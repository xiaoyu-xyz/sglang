//! Opaque values stored on radix-tree nodes.

use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

/// Backward-compatible value type when callers omit `V`.
#[cfg(feature = "torch")]
pub type DefaultRadixValue = tch::Tensor;

/// Torch-free default used by native consumers that omit `V`.
#[cfg(not(feature = "torch"))]
pub type DefaultRadixValue = PageValue<usize>;

/// Value operations required by the radix-tree mechanism.
///
/// Implementations should make [`Self::shallow_clone`] cheap. Values are index
/// descriptors rather than mutable KV data, so sharing immutable storage is
/// safe for CPU simulation backends.
pub trait RadixValue: Debug + Sized + 'static {
    /// Number of logical radix atoms represented by this value.
    fn len(&self) -> usize;

    /// Whether the value contains no atoms.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cheap handle clone used while collecting a matched path.
    fn shallow_clone(&self) -> Self;

    /// Independent copy used when the tree adopts externally-owned indices.
    fn deep_copy(&self) -> Self;

    /// View or copy a contiguous logical range.
    fn slice(&self, start: usize, len: usize) -> Self;

    /// Split an owned value into non-overlapping logical ranges.
    fn split_owned(self, at: usize) -> (Self, Self);

    /// Concatenate values in path order.
    fn concat(values: &[Self]) -> Self;

    /// Empty host-side value. Device-specific empty values are supplied to the
    /// tree constructor by the backend.
    fn empty() -> Self;

    /// Convert an index value to the integer representation required by SWA.
    fn to_i64(&self) -> Self;

    /// Add the leading singleton dimension required by Mamba transfers.
    fn unsqueeze_zero(&self) -> Self;

    /// Materialize integer values for optional inspection/debug APIs.
    fn to_i64_vec(&self) -> Vec<i64> {
        panic!("this value backend does not support integer inspection")
    }
}

/// Immutable, cheaply sliced list suitable for simulated KV page identifiers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageValue<T> {
    storage: Arc<[T]>,
    range: Range<usize>,
}

impl<T> PageValue<T> {
    pub fn from_vec(values: Vec<T>) -> Self {
        let len = values.len();
        Self {
            storage: values.into(),
            range: 0..len,
        }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.storage[self.range.clone()]
    }

    pub fn into_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        self.as_slice().to_vec()
    }
}

impl<T> Default for PageValue<T> {
    fn default() -> Self {
        Self {
            storage: Arc::from([]),
            range: 0..0,
        }
    }
}

impl<T> From<Vec<T>> for PageValue<T> {
    fn from(values: Vec<T>) -> Self {
        Self::from_vec(values)
    }
}

impl<T> RadixValue for PageValue<T>
where
    T: Clone + Debug + 'static,
{
    fn len(&self) -> usize {
        self.range.len()
    }

    fn shallow_clone(&self) -> Self {
        self.clone()
    }

    fn deep_copy(&self) -> Self {
        Self::from_vec(self.as_slice().to_vec())
    }

    fn slice(&self, start: usize, len: usize) -> Self {
        assert!(start <= self.len(), "slice start exceeds value length");
        assert!(
            len <= self.len() - start,
            "slice length exceeds value length"
        );
        let absolute_start = self.range.start + start;
        Self {
            storage: Arc::clone(&self.storage),
            range: absolute_start..absolute_start + len,
        }
    }

    fn split_owned(self, at: usize) -> (Self, Self) {
        assert!(at <= self.len(), "split point exceeds value length");
        let middle = self.range.start + at;
        let head = Self {
            storage: Arc::clone(&self.storage),
            range: self.range.start..middle,
        };
        let tail = Self {
            storage: self.storage,
            range: middle..self.range.end,
        };
        (head, tail)
    }

    fn concat(values: &[Self]) -> Self {
        let len = values.iter().map(Self::len).sum();
        let mut joined = Vec::with_capacity(len);
        for value in values {
            joined.extend_from_slice(value.as_slice());
        }
        Self::from_vec(joined)
    }

    fn empty() -> Self {
        Self::default()
    }

    fn to_i64(&self) -> Self {
        self.shallow_clone()
    }

    fn unsqueeze_zero(&self) -> Self {
        self.shallow_clone()
    }
}

#[cfg(feature = "torch")]
impl RadixValue for tch::Tensor {
    fn len(&self) -> usize {
        self.size()[0] as usize
    }

    fn shallow_clone(&self) -> Self {
        tch::Tensor::shallow_clone(self)
    }

    fn deep_copy(&self) -> Self {
        self.copy()
    }

    fn slice(&self, start: usize, len: usize) -> Self {
        self.narrow(0, start as i64, len as i64)
    }

    fn split_owned(self, at: usize) -> (Self, Self) {
        let len = RadixValue::len(&self);
        assert!(0 < at && at < len, "split point must be internal");
        (
            self.narrow(0, 0, at as i64).copy(),
            self.narrow(0, at as i64, (len - at) as i64).copy(),
        )
    }

    fn concat(values: &[Self]) -> Self {
        tch::Tensor::cat(values, 0)
    }

    fn empty() -> Self {
        tch::Tensor::empty([0], (tch::Kind::Int64, tch::Device::Cpu))
    }

    fn to_i64(&self) -> Self {
        self.to_kind(tch::Kind::Int64)
    }

    fn unsqueeze_zero(&self) -> Self {
        self.unsqueeze(0)
    }

    fn to_i64_vec(&self) -> Vec<i64> {
        Vec::<i64>::try_from(&self.to(tch::Device::Cpu)).expect("failed to copy radix value to CPU")
    }
}

#[cfg(test)]
mod tests {
    use super::{PageValue, RadixValue};

    #[test]
    fn page_value_slices_split_and_concatenates_without_changing_order() {
        let value = PageValue::from_vec(vec![10_u64, 11, 12, 13]);
        assert_eq!(value.slice(1, 2).as_slice(), &[11, 12]);

        let (head, tail) = value.split_owned(2);
        assert_eq!(head.as_slice(), &[10, 11]);
        assert_eq!(tail.as_slice(), &[12, 13]);
        assert_eq!(
            PageValue::concat(&[head, tail]).as_slice(),
            &[10, 11, 12, 13]
        );
    }
}
