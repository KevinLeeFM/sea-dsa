#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"

#include <memory>
#include <string>
#include <vector>

namespace seadsa {

class AbstractType {
public:
  enum class Kind { Bottom, Top, Int, Ptr, Seq, Prod, Sum };

  AbstractType() : m_kind(Kind::Top) {}
  explicit AbstractType(Kind K) : m_kind(K) {}

  static AbstractType bottom();
  static AbstractType top();
  static AbstractType intType();
  static AbstractType ptr(AbstractType pointee);
  static AbstractType seq(AbstractType element);
  static AbstractType prod(AbstractType left, AbstractType right);
  static AbstractType sum(std::vector<AbstractType> summands);

  Kind kind() const { return m_kind; }
  const AbstractType &pointee() const { return *m_pointee; }
  const AbstractType &left() const { return *m_left; }
  const AbstractType &right() const { return *m_right; }
  const std::vector<AbstractType> &summands() const { return m_summands; }

  std::string str() const;

  AbstractType(const AbstractType &) = default;
  AbstractType(AbstractType &&) = default;
  AbstractType &operator=(const AbstractType &) = default;
  AbstractType &operator=(AbstractType &&) = default;

private:
  Kind m_kind;
  std::shared_ptr<AbstractType> m_pointee;
  std::shared_ptr<AbstractType> m_left;
  std::shared_ptr<AbstractType> m_right;
  std::vector<AbstractType> m_summands;
};

struct TypeInferenceResult {
  llvm::DenseMap<const llvm::Value *, AbstractType> Types;
};

class TypeInferenceAnalysis
    : public llvm::AnalysisInfoMixin<TypeInferenceAnalysis> {
public:
  using Result = TypeInferenceResult;
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
  static llvm::AnalysisKey Key;
};

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
