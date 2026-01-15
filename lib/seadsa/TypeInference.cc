#include "seadsa/TypeInference.hh"

#include "seadsa/InitializePasses.hh"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>

#define DEBUG_TYPE "seadsa-typeinf"

using namespace llvm;
using namespace seadsa;

static cl::opt<bool> TypeInfLog(
    "sea-dsa-typeinf-log",
    cl::desc("Print TypeInference assignments per function to stderr"),
    cl::init(false));

namespace {

static void writeValueLabel(const Value &V, raw_ostream &OS) {
  if (V.hasName()) {
    OS << V.getName();
    return;
  }
  OS << "<unnamed>";
}

static void logType(const Function &F, const Value &V,
                    const TypeInferenceResult &R, AbsType Ty) {
  if (!llvm::DebugFlag && !TypeInfLog)
    return;

  SmallString<128> Msg;
  raw_svector_ostream OS(Msg);
  OS << F.getName() << ".";
  writeValueLabel(V, OS);
  OS << " -> ";
  R.Ctx->print(Ty, OS);

  LLVM_DEBUG(dbgs() << "[" << DEBUG_TYPE << "] " << OS.str() << "\n");
  if (TypeInfLog) {
    errs() << "[" << DEBUG_TYPE << "] " << OS.str() << "\n";
  }
}

/// Gather all integer types that are interchangeable with pointers for this
/// target, using the data layout (defaults from the fixed target triple).
///
/// Returned as a SmallVector instead of a single type to accommodate
/// architectures with heterogeneous pointer sizes across address spaces. Most
/// targets yield one entry (default pointer width), but the interface remains
/// extensible to multiple widths.
static SmallVector<llvm::Type *> pointerInterchangeableIntegers(
    const DataLayout &DL, LLVMContext &Ctx) {
  SmallVector<llvm::Type *> Result;
  unsigned Width = DL.getPointerSizeInBits(0);
  Result.push_back(IntegerType::get(Ctx, Width));
  return Result;
}

} // namespace

//===----------------------------------------------------------------------===
// LLVM Type -> AbstractType conversion
//===----------------------------------------------------------------------===

AbsType seadsa::convertLLVMType(const llvm::Type &Ty, const DataLayout &DL,
                                AbstractTypeContext &Ctx) {
  if (Ty.isVoidTy()) return Ctx.mkBottom();

  LLVMContext &LLVMCtx = Ty.getContext();
  SmallVector<llvm::Type *> PtrLikeInts = pointerInterchangeableIntegers(DL, LLVMCtx);
  SmallVector<AbsType, 4> PtrLikeAbstract;
  for (llvm::Type *ITy : PtrLikeInts) {
    PtrLikeAbstract.push_back(Ctx.mkScalar(ITy));
  }

  auto ptrLikeSum = [&](AbsType PtrTop) {
    SmallVector<AbsType, 8> Parts(PtrLikeAbstract.begin(), PtrLikeAbstract.end());
    Parts.push_back(PtrTop);
    return Ctx.mkSum(Parts);
  };

  if (Ty.isPointerTy()) {
    AbsType PtrTop = Ctx.mkPtr(Ctx.mkTop());
    return ptrLikeSum(PtrTop);
  }

  if (Ty.isIntegerTy()) {
    unsigned BW = Ty.getIntegerBitWidth();
    bool IsPtrSized = llvm::is_contained(PtrLikeInts, &Ty) ||
                      llvm::any_of(PtrLikeInts, [&](llvm::Type *ITy) {
                        return cast<IntegerType>(ITy)->getBitWidth() == BW;
                      });
    if (IsPtrSized) {
      AbsType PtrTop = Ctx.mkPtr(Ctx.mkTop());
      return ptrLikeSum(PtrTop);
    }
    return Ctx.mkScalar(&Ty);
  }

  if (Ty.isFloatingPointTy()) {
    return Ctx.mkScalar(&Ty);
  }

  if (Ty.isStructTy()) {
    const StructType *ST = cast<StructType>(&Ty);
    if (ST->isOpaque()) return Ctx.mkTop();
    SmallVector<AbsType, 8> Fields;
    for (llvm::Type *Elt : ST->elements()) {
      Fields.push_back(convertLLVMType(*Elt, DL, Ctx));
    }
    return Ctx.mkProd(Fields);
  }

  if (Ty.isArrayTy()) {
    // No LLVM type maps to Seq; conservatively treat as Top.
    return Ctx.mkTop();
  }

  if (Ty.isVectorTy()) {
    return Ctx.mkTop();
  }

  // Fallback to Top for function, metadata, label, etc.
  return Ctx.mkTop();
}

//===----------------------------------------------------------------------===
// Type inference driver
//===----------------------------------------------------------------------===

namespace {

TypeInferenceResult inferTypes(Function &F) {
  TypeInferenceResult R;
  R.Ctx = std::make_shared<AbstractTypeContext>();

  const DataLayout &DL = F.getParent()->getDataLayout();

  for (Argument &Arg : F.args()) {
    AbsType Ty = convertLLVMType(*Arg.getType(), DL, *R.Ctx);
    R.Types[&Arg] = Ty;
    logType(F, Arg, R, Ty);
  }

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (!I.getType()->isVoidTy()) {
        AbsType Ty = convertLLVMType(*I.getType(), DL, *R.Ctx);
        R.Types[&I] = Ty;
        logType(F, I, R, Ty);
      }
    }
  }

  return R;
}

} // namespace

TypeInferenceAnalysis::Result
TypeInferenceAnalysis::run(Function &F, FunctionAnalysisManager &) {
  return inferTypes(F);
}

AnalysisKey TypeInferenceAnalysis::Key;

char TypeInferenceWrapperPass::ID = 0;

TypeInferenceWrapperPass::TypeInferenceWrapperPass() : FunctionPass(ID) {}

bool TypeInferenceWrapperPass::runOnFunction(Function &F) {
  m_result = inferTypes(F);
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
                      "SeaDsa Type Inference", false, true)
INITIALIZE_PASS_END(TypeInferenceWrapperPass, "seadsa-typeinf",
                    "SeaDsa Type Inference", false, true)
