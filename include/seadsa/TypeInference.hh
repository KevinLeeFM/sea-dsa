#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
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

/// Convert an LLVM type into an AbstractType owned by the given context.
///
/// Pointer and pointer-interchangeable scalars map to a canonical Sum that
/// includes Ptr(Top); structs map to Prod recursively; no LLVM type maps to
/// Seq; unhandled kinds conservatively map to Top.
AbsType convertLLVMType(const llvm::Type &Ty, const llvm::DataLayout &DL,
                     AbstractTypeContext &Ctx);

/// Context that uniques and owns all AbstractType nodes.
///
/// Provides canonical builders, interning, and deterministic stringification.
/// Construction is guarded against cycles to preserve termination guarantees.
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

private:
  struct Key;
  struct KeyInfo;
  AbsType uniquedInsert(Key K);
  bool formsCycle(AbsType Candidate, llvm::ArrayRef<AbsType> Children) const;

  llvm::DenseMap<Key, std::unique_ptr<TypeNode>, KeyInfo> m_cache;
  AbsType m_bottom;
  AbsType m_top;
};

/// Function-level mapping from SSA values to abstract types.
/// The shared context keeps all type nodes alive for the lifetime of the
/// result.
struct TypeInferenceResult {
  std::shared_ptr<AbstractTypeContext> Ctx;
  llvm::DenseMap<const llvm::Value *, AbsType> Types;
};

/// New-PM analysis that assigns an abstract type to every SSA value.
///
/// The result maps SSA values to canonical AbstractTypes owned by the shared
/// context.
class TypeInferenceAnalysis
    : public llvm::AnalysisInfoMixin<TypeInferenceAnalysis> {
public:
  using Result = TypeInferenceResult;
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
  static llvm::AnalysisKey Key;
};

/// Legacy FunctionPass wrapper to surface TypeInference results.
///
/// Provides access to the same canonicalized mapping when using the legacy
/// pass manager.
class TypeInferenceWrapperPass : public llvm::FunctionPass {
public:
  static char ID;
  TypeInferenceWrapperPass();

  bool runOnFunction(llvm::Function &F) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  TypeInferenceResult &getResult() { return m_result; }
  const TypeInferenceResult &getResult() const { return m_result; }

private:
  TypeInferenceResult m_result;
};

llvm::FunctionPass *createTypeInferenceWrapperPass();

/// Register the analysis with the new pass manager.
void registerTypeInferenceAnalysis(llvm::PassBuilder &PB);

} // namespace seadsa
