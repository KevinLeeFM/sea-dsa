#pragma once

#include "seadsa/AbstractType.hh"

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"

#include <memory>

namespace seadsa {

/// Convert an LLVM type into an AbstractType owned by the given context.
///
/// Pointer and pointer-interchangeable scalars map to a canonical Sum that
/// includes Ptr(Top); structs map to Prod recursively; no LLVM type maps to
/// Seq; unhandled kinds conservatively map to Top.
AbsType convertLLVMType(const llvm::Type &Ty, const llvm::DataLayout &DL,
                        AbstractTypeContext &Ctx);

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
