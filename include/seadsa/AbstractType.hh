#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

namespace seadsa {

class AbstractTypeContext;

/// Immutable node in the AbstractType graph.
///
/// All nodes are owned and uniqued by an AbstractTypeContext. Children store
/// raw pointers to other nodes also owned by the same context. Nodes are
/// immutable after construction.
class TypeNode {
public:
  enum class Kind { Bottom, Top, Scalar, Ptr, Seq, Prod, Sum };

  Kind K;
  const llvm::Type *ScalarTy = nullptr;
  llvm::SmallVector<const TypeNode *, 4> Children;
};

using AbsType = const TypeNode *;

/// Lightweight handle to an abstract type owned by an AbstractTypeContext.
class AbstractType {
public:
  using Kind = TypeNode::Kind;
  AbstractType() = default;
  explicit AbstractType(AbsType T) : m_type(T) {}

  /// Return the discriminant for this type. Defaults to Top when null.
  Kind kind() const;
  AbsType raw() const { return m_type; }
  bool isValid() const { return m_type != nullptr; }

  /// For Ptr/Seq returns the sole child; otherwise nullptr.
  AbsType pointee() const;
  /// Return the ordered children for Prod/Sum; empty otherwise.
  llvm::ArrayRef<AbsType> elements() const;
  /// Best-effort printer without requiring a context (for debugging only).
  void print(llvm::raw_ostream &OS) const;

  /// Logical structural equality check (ignores interning identity).
  bool equals(const AbstractType &Other) const;

private:
  AbsType m_type = nullptr;
};

/// Context that uniques and owns all AbstractType nodes.
///
/// Provides canonical builders, interning, deterministic stringification, and
/// lattice operations derived from the IntraprocMemIR type spec.
class AbstractTypeContext {
public:
  AbstractTypeContext();

  AbsType mkBottom();
  AbsType mkTop();
  AbsType mkScalar(const llvm::Type *Ty);
  AbsType mkPtr(AbsType Pointee);
  AbsType mkSeq(AbsType Element);
  AbsType mkProd(llvm::ArrayRef<AbsType> Elements);
  AbsType mkSum(llvm::ArrayRef<AbsType> Summands);
  AbsType join(AbsType A, AbsType B);

  bool equals(AbsType A, AbsType B) const;
  /// Produce a deterministic textual representation for the given type.
  void print(AbsType T, llvm::raw_ostream &OS) const;

  /// Normalize under the lattice equivalence rules (≈ in the spec).
  AbsType normalize(AbsType T);

  /// Lattice ordering (≤ in the spec).
  bool leq(AbsType A, AbsType B);

  /// Equivalence under normalization.
  bool equivalent(AbsType A, AbsType B);

  /// Greatest lower bound of two types.
  AbsType meetTwo(AbsType A, AbsType B);

  /// Greatest lower bound of a sequence (empty -> Top).
  AbsType meet(llvm::ArrayRef<AbsType> Types);

  /// Dereference (⊥ when undefined).
  AbsType deref(AbsType T);

  /// Truncation (suffix from index, per spec).
  AbsType trunc(AbsType T, unsigned Index);

  /// Return true if T may contain a pointer anywhere inside.
  /// Prototype helper intended for upcoming SeaDSA integration.
  bool mayContainPointer(AbsType T);

private:
  struct Key {
    TypeNode::Kind K = TypeNode::Kind::Top;
    const llvm::Type *Scalar = nullptr;
    llvm::SmallVector<AbsType, 4> Children;
  };

  struct KeyInfo {
    static Key getEmptyKey() {
      Key K;
      K.Scalar = reinterpret_cast<llvm::Type *>(1);
      return K;
    }
    static Key getTombstoneKey() {
      Key K;
      K.Scalar = reinterpret_cast<llvm::Type *>(2);
      return K;
    }
    static unsigned getHashValue(const Key &K) {
      using llvm::hash_combine;
      using llvm::hash_combine_range;
      unsigned H = hash_combine(static_cast<unsigned>(K.K), K.Scalar);
      return hash_combine(H, hash_combine_range(K.Children.begin(),
                                                K.Children.end()));
    }
    static bool isEqual(const Key &LHS, const Key &RHS) {
      if (LHS.Scalar != RHS.Scalar) return false;
      if (LHS.K != RHS.K) return false;
      if (LHS.Children.size() != RHS.Children.size()) return false;
      for (size_t I = 0, E = LHS.Children.size(); I != E; ++I) {
        if (LHS.Children[I] != RHS.Children[I]) return false;
      }
      return true;
    }
  };

  AbsType uniquedInsert(Key K);
  bool formsCycle(AbsType Candidate, llvm::ArrayRef<AbsType> Children) const;

  // Helpers for normalization and canonicalization.
  AbsType normalizeSeq(AbsType Element);
  AbsType normalizeProdChildren(llvm::ArrayRef<AbsType> Elements);
  AbsType normalizeSum(llvm::ArrayRef<AbsType> Summands);

  llvm::DenseMap<Key, std::unique_ptr<TypeNode>, KeyInfo> m_cache;
  AbsType m_bottom;
  AbsType m_top;
};

} // namespace seadsa
