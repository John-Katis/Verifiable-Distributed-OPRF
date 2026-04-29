use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type PartyId = usize;

/// Communication statistics from a simulated network.
///
/// Server-to-server traffic is split into `p2p_bytes` (point-to-point) and
/// `broadcast_bytes` (one sender to all parties). Server-to-client traffic
/// is tracked separately in `client_bytes` so that the bench can report the
/// S↔S vs S↔C split directly from the protocol's own bookkeeping instead
/// of recomputing it.
#[derive(Clone, Debug, Default)]
pub struct CommStats {
    pub p2p_bytes: usize,
    pub p2p_messages: usize,
    pub broadcast_bytes: usize,
    pub broadcast_messages: usize,
    pub rounds: usize,
    /// Total bytes sent from any server to the client.
    pub client_bytes: usize,
    /// Number of server→client messages.
    pub client_messages: usize,
}

impl CommStats {
    /// Server↔server bytes only (p2p + broadcast). Client-facing bytes are
    /// deliberately excluded — they live in `client_bytes`.
    pub fn total_bytes(&self) -> usize {
        self.p2p_bytes + self.broadcast_bytes
    }

    /// Merge stats from a sequential step (rounds add up).
    pub fn merge(&mut self, other: &CommStats) {
        self.p2p_bytes += other.p2p_bytes;
        self.p2p_messages += other.p2p_messages;
        self.broadcast_bytes += other.broadcast_bytes;
        self.broadcast_messages += other.broadcast_messages;
        self.rounds += other.rounds;
        self.client_bytes += other.client_bytes;
        self.client_messages += other.client_messages;
    }

    /// Merge stats from parallel operations within the same round
    /// (bytes add up, rounds take the max).
    pub fn merge_parallel(&mut self, other: &CommStats) {
        self.p2p_bytes += other.p2p_bytes;
        self.p2p_messages += other.p2p_messages;
        self.broadcast_bytes += other.broadcast_bytes;
        self.broadcast_messages += other.broadcast_messages;
        if other.rounds > self.rounds {
            self.rounds = other.rounds;
        }
        self.client_bytes += other.client_bytes;
        self.client_messages += other.client_messages;
    }
}

/// A message sent over the network.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub sender: PartyId,
    pub data: Vec<u8>,
}

/// Simulated synchronous network for n parties.
/// Supports round-based broadcast and point-to-point messaging.
pub struct SimulatedNetwork {
    pub n_parties: usize,
    current_round: usize,
    /// broadcast_buf[round] = list of (sender, data)
    broadcast_buf: HashMap<usize, Vec<Message>>,
    /// p2p_buf[round][(sender, receiver)] = data. Overwritten on repeat;
    /// readers like `rss_mul` expect exactly one payload per pair per round.
    p2p_buf: HashMap<usize, HashMap<(PartyId, PartyId), Vec<u8>>>,
    /// Byte accounting that accumulates across all same-round sends between
    /// the same (sender, receiver). Protocols that FS-batch multiple VSS
    /// rounds into one synchronous round (e.g., `vip_single` γ iterations of
    /// q(1)/q(2)/q(3) followed by end-of-loop a₁/b₁) need every transmitted
    /// byte counted, but each individual payload is semantically independent
    /// — `get_p2p` would conflate them. We keep payload storage as-is (last
    /// write wins) for readers that rely on it, and accumulate lengths +
    /// message counts here for `total_p2p_bytes` / `total_p2p_messages`.
    p2p_bytes_counter: HashMap<usize, HashMap<(PartyId, PartyId), usize>>,
    p2p_messages_counter: HashMap<usize, HashMap<(PartyId, PartyId), usize>>,
    /// client_buf[round] = list of server→client messages (server is sender).
    client_buf: HashMap<usize, Vec<Message>>,
}

impl SimulatedNetwork {
    pub fn new(n_parties: usize) -> Self {
        SimulatedNetwork {
            n_parties,
            current_round: 0,
            broadcast_buf: HashMap::new(),
            p2p_buf: HashMap::new(),
            p2p_bytes_counter: HashMap::new(),
            p2p_messages_counter: HashMap::new(),
            client_buf: HashMap::new(),
        }
    }

    pub fn current_round(&self) -> usize {
        self.current_round
    }

    /// Advance to the next round.
    pub fn next_round(&mut self) {
        self.current_round += 1;
    }

    /// Broadcast a message to all parties.
    pub fn broadcast(&mut self, sender: PartyId, data: Vec<u8>) {
        self.broadcast_buf
            .entry(self.current_round)
            .or_default()
            .push(Message {
                sender,
                data,
            });
    }

    /// Send a point-to-point message.
    ///
    /// `p2p_buf` retains only the last payload for `(sender, receiver)` in a
    /// given round — this matches what `rss_mul`-style readers expect. Byte
    /// accounting goes into `p2p_bytes_counter`, which accumulates the full
    /// traffic (needed so that FS-batched VSS rounds like `vip_single`'s γ
    /// iterations are counted in full rather than collapsing to the final
    /// send).
    pub fn send_p2p(&mut self, sender: PartyId, receiver: PartyId, data: Vec<u8>) {
        let len = data.len();
        *self
            .p2p_bytes_counter
            .entry(self.current_round)
            .or_default()
            .entry((sender, receiver))
            .or_insert(0) += len;
        *self
            .p2p_messages_counter
            .entry(self.current_round)
            .or_default()
            .entry((sender, receiver))
            .or_insert(0) += 1;
        self.p2p_buf
            .entry(self.current_round)
            .or_default()
            .insert((sender, receiver), data);
    }

    /// Send a message from a server to the (external, non-numbered) client.
    /// The client is not a party in `SimulatedNetwork`, so these bytes are
    /// tracked in a separate bucket (`client_bytes` in [`CommStats`]).
    pub fn send_to_client(&mut self, sender: PartyId, data: Vec<u8>) {
        self.client_buf
            .entry(self.current_round)
            .or_default()
            .push(Message { sender, data });
    }

    /// Get all broadcast messages for a given round.
    pub fn get_broadcasts(&self, round: usize) -> Vec<&Message> {
        self.broadcast_buf
            .get(&round)
            .map(|msgs| msgs.iter().collect())
            .unwrap_or_default()
    }

    /// Get all broadcast messages for a given round from a specific sender.
    pub fn get_broadcast_from(&self, round: usize, sender: PartyId) -> Option<&Message> {
        self.broadcast_buf.get(&round).and_then(|msgs| {
            msgs.iter().find(|m| m.sender == sender)
        })
    }

    /// Get a P2P message for a given round.
    pub fn get_p2p(&self, round: usize, sender: PartyId, receiver: PartyId) -> Option<&Vec<u8>> {
        self.p2p_buf
            .get(&round)
            .and_then(|map| map.get(&(sender, receiver)))
    }

    /// Reset the network for reuse.
    pub fn reset(&mut self) {
        self.current_round = 0;
        self.broadcast_buf.clear();
        self.p2p_buf.clear();
        self.p2p_bytes_counter.clear();
        self.p2p_messages_counter.clear();
        self.client_buf.clear();
    }

    /// Total bytes sent from any server to the client across all rounds.
    pub fn total_client_bytes(&self) -> usize {
        self.client_buf
            .values()
            .flat_map(|msgs| msgs.iter())
            .map(|m| m.data.len())
            .sum()
    }

    /// Total number of server→client messages.
    pub fn total_client_messages(&self) -> usize {
        self.client_buf.values().map(|msgs| msgs.len()).sum()
    }

    /// Total broadcast bytes sent across all rounds.
    pub fn total_broadcast_bytes(&self) -> usize {
        self.broadcast_buf.values()
            .flat_map(|msgs| msgs.iter())
            .map(|m| m.data.len())
            .sum()
    }

    /// Total P2P bytes sent across all rounds (accumulated — repeated
    /// same-round sends between the same pair are summed, not overwritten).
    pub fn total_p2p_bytes(&self) -> usize {
        self.p2p_bytes_counter
            .values()
            .flat_map(|map| map.values())
            .sum()
    }

    /// Total number of broadcast messages.
    pub fn total_broadcast_messages(&self) -> usize {
        self.broadcast_buf.values().map(|msgs| msgs.len()).sum()
    }

    /// Total number of P2P messages (accumulated — one per `send_p2p` call).
    pub fn total_p2p_messages(&self) -> usize {
        self.p2p_messages_counter
            .values()
            .flat_map(|map| map.values())
            .sum()
    }

    /// Number of communication rounds used.
    pub fn num_rounds(&self) -> usize {
        self.current_round + 1
    }

    /// Extract communication statistics.
    pub fn stats(&self) -> CommStats {
        CommStats {
            p2p_bytes: self.total_p2p_bytes(),
            p2p_messages: self.total_p2p_messages(),
            broadcast_bytes: self.total_broadcast_bytes(),
            broadcast_messages: self.total_broadcast_messages(),
            rounds: self.num_rounds(),
            client_bytes: self.total_client_bytes(),
            client_messages: self.total_client_messages(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast() {
        let mut net = SimulatedNetwork::new(3);
        net.broadcast(0, vec![1, 2, 3]);
        net.broadcast(1, vec![4, 5, 6]);

        let msgs = net.get_broadcasts(0);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].sender, 0);
        assert_eq!(msgs[0].data, vec![1, 2, 3]);
    }

    #[test]
    fn test_p2p() {
        let mut net = SimulatedNetwork::new(3);
        net.send_p2p(0, 1, vec![10, 20]);
        net.send_p2p(0, 2, vec![30, 40]);

        assert_eq!(net.get_p2p(0, 0, 1), Some(&vec![10, 20]));
        assert_eq!(net.get_p2p(0, 0, 2), Some(&vec![30, 40]));
        assert_eq!(net.get_p2p(0, 1, 0), None);
    }

    #[test]
    fn test_p2p_bytes_accumulate_across_same_round_repeats() {
        let mut net = SimulatedNetwork::new(3);
        // Simulate `vip_single` γ=4 iterations of 30-byte VSS between the
        // same (sender, receiver) within a single FS-batched round, plus a
        // smaller end-of-loop send. Payload overwrites (matching `get_p2p`
        // semantics), but bytes and message counts accumulate.
        net.send_p2p(0, 1, vec![0u8; 30]);
        net.send_p2p(0, 1, vec![0u8; 30]);
        net.send_p2p(0, 1, vec![0u8; 30]);
        net.send_p2p(0, 1, vec![0u8; 30]);
        net.send_p2p(0, 1, vec![0u8; 10]);
        assert_eq!(net.total_p2p_bytes(), 130);
        assert_eq!(net.total_p2p_messages(), 5);
        // Latest payload still wins for get_p2p readers.
        assert_eq!(net.get_p2p(0, 0, 1), Some(&vec![0u8; 10]));
    }

    #[test]
    fn test_rounds() {
        let mut net = SimulatedNetwork::new(3);
        net.broadcast(0, vec![1]);
        net.next_round();
        net.broadcast(1, vec![2]);

        assert_eq!(net.get_broadcasts(0).len(), 1);
        assert_eq!(net.get_broadcasts(1).len(), 1);
        assert_eq!(net.get_broadcasts(0)[0].sender, 0);
        assert_eq!(net.get_broadcasts(1)[0].sender, 1);
    }
}
