use crate::error::{PandasError, Result};

/// Row index for DataFrame and Series.
#[derive(Debug, Clone)]
pub enum Index {
    Range(RangeIndex),
    Int64(Vec<i64>),
    Str(Vec<String>),
}

/// A range-based index (0..n), no allocation needed.
#[derive(Debug, Clone)]
pub struct RangeIndex {
    pub start: i64,
    pub stop: i64,
    pub step: i64,
}

impl RangeIndex {
    pub fn new(len: usize) -> Self {
        RangeIndex {
            start: 0,
            stop: len as i64,
            step: 1,
        }
    }

    pub fn len(&self) -> usize {
        if self.step > 0 {
            ((self.stop - self.start + self.step - 1) / self.step).max(0) as usize
        } else if self.step < 0 {
            ((self.start - self.stop - self.step - 1) / (-self.step)).max(0) as usize
        } else {
            0
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, idx: usize) -> Option<i64> {
        if idx < self.len() {
            Some(self.start + (idx as i64) * self.step)
        } else {
            None
        }
    }
}

impl Index {
    pub fn len(&self) -> usize {
        match self {
            Index::Range(r) => r.len(),
            Index::Int64(v) => v.len(),
            Index::Str(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a default RangeIndex(0..len).
    pub fn default_range(len: usize) -> Self {
        Index::Range(RangeIndex::new(len))
    }

    /// Select rows by positional indices.
    pub fn take(&self, indices: &[usize]) -> Result<Index> {
        match self {
            Index::Range(r) => {
                let values: Result<Vec<i64>> = indices
                    .iter()
                    .map(|&i| {
                        r.get(i).ok_or_else(|| {
                            PandasError::IndexError(format!(
                                "index {} out of bounds for length {}",
                                i,
                                r.len()
                            ))
                        })
                    })
                    .collect();
                Ok(Index::Int64(values?))
            }
            Index::Int64(v) => {
                let values: Result<Vec<i64>> = indices
                    .iter()
                    .map(|&i| {
                        v.get(i).copied().ok_or_else(|| {
                            PandasError::IndexError(format!(
                                "index {} out of bounds for length {}",
                                i,
                                v.len()
                            ))
                        })
                    })
                    .collect();
                Ok(Index::Int64(values?))
            }
            Index::Str(v) => {
                let values: Result<Vec<String>> = indices
                    .iter()
                    .map(|&i| {
                        v.get(i).cloned().ok_or_else(|| {
                            PandasError::IndexError(format!(
                                "index {} out of bounds for length {}",
                                i,
                                v.len()
                            ))
                        })
                    })
                    .collect();
                Ok(Index::Str(values?))
            }
        }
    }

    pub fn get_loc_int(&self, label: i64) -> Option<usize> {
        match self {
            Index::Range(r) => {
                if r.step == 0 {
                    return None;
                }
                let offset = label - r.start;
                if offset % r.step != 0 {
                    return None;
                }
                let pos = (offset / r.step) as usize;
                if pos < r.len() {
                    Some(pos)
                } else {
                    None
                }
            }
            Index::Int64(v) => v.iter().position(|&x| x == label),
            Index::Str(_) => None,
        }
    }

    pub fn get_loc_str(&self, label: &str) -> Option<usize> {
        match self {
            Index::Str(v) => v.iter().position(|x| x == label),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_index() {
        let r = RangeIndex::new(5);
        assert_eq!(r.len(), 5);
        assert_eq!(r.get(0), Some(0));
        assert_eq!(r.get(4), Some(4));
        assert_eq!(r.get(5), None);
    }

    #[test]
    fn test_index_default_range() {
        let idx = Index::default_range(3);
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn test_index_take() {
        let idx = Index::Int64(vec![10, 20, 30, 40]);
        let taken = idx.take(&[3, 0]).unwrap();
        match taken {
            Index::Int64(v) => assert_eq!(v, vec![40, 10]),
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_index_get_loc_range() {
        let idx = Index::default_range(5);
        assert_eq!(idx.get_loc_int(0), Some(0));
        assert_eq!(idx.get_loc_int(4), Some(4));
        assert_eq!(idx.get_loc_int(5), None);
    }

    #[test]
    fn test_index_get_loc_str() {
        let idx = Index::Str(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(idx.get_loc_str("b"), Some(1));
        assert_eq!(idx.get_loc_str("z"), None);
    }
}
