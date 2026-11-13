//! Bounded mobile command delivery. Only producer threads take a mutex; the audio receiver
//! and the reverse retirement channel use their own single-owner ring endpoints directly.
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::TrySendError;
use std::sync::{Arc, Mutex, PoisonError};

pub(super) struct Sender<T> {
    producer: Mutex<rtrb::Producer<T>>,
    overflowed: Arc<AtomicBool>,
}
pub(super) struct Receiver<T> {
    consumer: rtrb::Consumer<T>,
    overflowed: Arc<AtomicBool>,
}
pub(super) struct SendError<T>(T);
impl<T> fmt::Debug for SendError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("mobile command queue is full or disconnected")
    }
}

pub(super) fn channel<T>() -> (Sender<T>, Receiver<T>) {
    channel_with_capacity(8)
}
pub(super) fn channel_with_capacity<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    let (producer, consumer) = rtrb::RingBuffer::new(capacity);
    let overflowed = Arc::new(AtomicBool::new(false));
    (Sender { producer: Mutex::new(producer), overflowed: Arc::clone(&overflowed) }, Receiver { consumer, overflowed })
}
impl<T> Sender<T> {
    pub(super) fn send(&self, value: T) -> Result<(), SendError<T>> {
        let mut producer = self.producer.lock().unwrap_or_else(PoisonError::into_inner);
        if producer.is_abandoned() {
            return Err(SendError(value));
        }
        producer.push(value).map_err(|rtrb::PushError::Full(value)| {
            self.overflowed.store(true, Ordering::Release);
            SendError(value)
        })
    }
    pub(super) fn overflowed(&self) -> bool {
        self.overflowed.load(Ordering::Acquire)
    }
}
impl<T> Receiver<T> {
    pub(super) fn try_recv(&mut self) -> Result<T, rtrb::PopError> {
        self.consumer.pop()
    }
    pub(super) fn overflowed(&self) -> bool {
        self.overflowed.load(Ordering::Acquire)
    }
}

pub(super) struct ReturnSender<T>(rtrb::Producer<T>);
pub(super) struct ReturnReceiver<T>(rtrb::Consumer<T>);
pub(super) fn return_channel<T>(capacity: usize) -> (ReturnSender<T>, ReturnReceiver<T>) {
    let (producer, consumer) = rtrb::RingBuffer::new(capacity);
    (ReturnSender(producer), ReturnReceiver(consumer))
}
impl<T> ReturnSender<T> {
    pub(super) fn try_send(&mut self, value: T) -> Result<(), TrySendError<T>> {
        if self.0.is_abandoned() {
            return Err(TrySendError::Disconnected(value));
        }
        self.0.push(value).map_err(|rtrb::PushError::Full(value)| TrySendError::Full(value))
    }
}
impl<T> ReturnReceiver<T> {
    pub(super) fn try_recv(&mut self) -> Result<T, rtrb::PopError> {
        self.0.pop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_can_pop_while_the_producer_mutex_is_held() {
        let (sender, mut receiver) = channel_with_capacity(2);
        sender.send(10).unwrap();
        let _producer_busy = sender.producer.lock().unwrap();
        assert_eq!(receiver.try_recv(), Ok(10));
        assert_eq!(receiver.try_recv(), Err(rtrb::PopError::Empty));
    }

    #[test]
    fn full_command_queue_keeps_order_and_latches_overload() {
        let (sender, mut receiver) = channel_with_capacity(2);
        sender.send(10).unwrap();
        sender.send(20).unwrap();
        assert!(!sender.overflowed());
        assert!(sender.send(30).is_err());
        assert!(receiver.overflowed());
        assert_eq!(receiver.try_recv(), Ok(10));
        assert_eq!(receiver.try_recv(), Ok(20));
        assert_eq!(receiver.try_recv(), Err(rtrb::PopError::Empty));
    }

    #[test]
    fn playback_command_ring_absorbs_a_burst_of_live_midi_batches() {
        let (sender, mut receiver) = channel_with_capacity(crate::COMMAND_QUEUE_CAPACITY);
        for batch in 0..48 {
            sender.send(batch).unwrap();
        }
        assert!(!sender.overflowed());
        for batch in 0..48 {
            assert_eq!(receiver.try_recv(), Ok(batch));
        }
    }

    #[test]
    fn full_return_ring_retains_ownership_in_the_error() {
        let (mut sender, mut receiver) = return_channel(1);
        sender.try_send(Arc::new(10)).unwrap();
        let value = Arc::new(20);
        let retained = match sender.try_send(Arc::clone(&value)) {
            Err(TrySendError::Full(retained)) => retained,
            _ => panic!("expected full return ring"),
        };
        assert_eq!(Arc::strong_count(&value), 2);
        assert_eq!(*receiver.try_recv().unwrap(), 10);
        sender.try_send(retained).unwrap();
        assert_eq!(*receiver.try_recv().unwrap(), 20);
        assert_eq!(Arc::strong_count(&value), 1);
    }
}
