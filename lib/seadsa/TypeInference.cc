#include "seadsa/TypeInference.hh"

#include "seadsa/InitializePasses.hh"

#include "llvm/InitializePasses.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "seadsa-typeinf"

using namespace llvm;
using namespace seadsa;

static cl::opt<bool> TypeInfLog(
  "sea-dsa-typeinf-log",
  cl::desc("Print TypeInference Top-counts per function to stderr"),
  cl::init(false));

namespace {

std::string valueName(const Value &V) {
  if (V.hasName()) return V.getName().str();
  std::string S;
  raw_string_ostream OS(S);
  V.printAsOperand(OS, false);
  return OS.str();
}

void logType(const Function &F, const Value &V, const AbstractType &Ty) {
  auto Msg = [&]() {
    std::string S;
    raw_string_ostream OS(S);
    OS << F.getName() << "." << valueName(V) << " -> " << Ty.str();
    return OS.str();
  }();

  LLVM_DEBUG(dbgs() << "[" << DEBUG_TYPE << "] " << Msg << "\n");
  if (TypeInfLog) {
    errs() << "[" << DEBUG_TYPE << "] " << Msg << "\n";
  }
}

} // namespace

AbstractType AbstractType::bottom() { return AbstractType(Kind::Bottom); }

AbstractType AbstractType::top() { return AbstractType(Kind::Top); }

AbstractType AbstractType::intType() { return AbstractType(Kind::Int); }

AbstractType AbstractType::ptr(AbstractType pointee) {
  AbstractType T(Kind::Ptr);
  T.m_pointee = std::make_shared<AbstractType>(std::move(pointee));
  return T;
}

AbstractType AbstractType::seq(AbstractType element) {
  AbstractType T(Kind::Seq);
  T.m_pointee = std::make_shared<AbstractType>(std::move(element));
  return T;
}

AbstractType AbstractType::prod(AbstractType left, AbstractType right) {
  AbstractType T(Kind::Prod);
  T.m_left = std::make_shared<AbstractType>(std::move(left));
  T.m_right = std::make_shared<AbstractType>(std::move(right));
  return T;
}

AbstractType AbstractType::sum(std::vector<AbstractType> summands) {
  AbstractType T(Kind::Sum);
  T.m_summands = std::move(summands);
  return T;
}

std::string AbstractType::str() const {
  switch (m_kind) {
  case Kind::Bottom:
    return "⊥";
  case Kind::Top:
    return "⊤";
  case Kind::Int:
    return "Int";
  case Kind::Ptr:
    return std::string("Ptr(") + m_pointee->str() + ")";
  case Kind::Seq:
    return std::string("Seq(") + m_pointee->str() + ")";
  case Kind::Prod:
    return std::string("(") + m_left->str() + " × " + m_right->str() + ")";
  case Kind::Sum: {
    if (m_summands.empty()) return "⊥";
    std::string S;
    for (size_t i = 0; i < m_summands.size(); ++i) {
      if (i) S += " + ";
      S += m_summands[i].str();
    }
    return S;
  }
  }
  return "⊤";
}

namespace {
TypeInferenceResult inferAllTop(Function &F) {
  TypeInferenceResult R;

  for (Argument &Arg : F.args()) {
    auto Ty = AbstractType::top();
    R.Types[&Arg] = Ty;
    logType(F, Arg, Ty);
  }

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (!I.getType()->isVoidTy()) {
        auto Ty = AbstractType::top();
        R.Types[&I] = Ty;
        logType(F, I, Ty);
      }
    }
  }
  return R;
}
} // namespace

TypeInferenceAnalysis::Result
TypeInferenceAnalysis::run(Function &F, FunctionAnalysisManager &) {
  return inferAllTop(F);
}

AnalysisKey TypeInferenceAnalysis::Key;

char TypeInferenceWrapperPass::ID = 0;

TypeInferenceWrapperPass::TypeInferenceWrapperPass() : FunctionPass(ID) {}

bool TypeInferenceWrapperPass::runOnFunction(Function &F) {
  m_result = inferAllTop(F);
  return false;
}

void TypeInferenceWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

FunctionPass *seadsa::createTypeInferenceWrapperPass() {
  return new TypeInferenceWrapperPass();
}

void seadsa::registerTypeInferenceAnalysis(PassBuilder &PB) {
  PB.registerAnalysisRegistrationCallback(
      [](FunctionAnalysisManager &FAM) {
        FAM.registerPass([] { return TypeInferenceAnalysis(); });
      });
}

INITIALIZE_PASS_BEGIN(TypeInferenceWrapperPass, "seadsa-typeinf",
                      "SeaDsa Type Inference (Top)", false, true)
INITIALIZE_PASS_END(TypeInferenceWrapperPass, "seadsa-typeinf",
                    "SeaDsa Type Inference (Top)", false, true)
