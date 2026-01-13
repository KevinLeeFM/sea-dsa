#include "seadsa/TypeInference.hh"

#include "seadsa/InitializePasses.hh"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
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

#define DEBUG_TYPE "seadsa-typeinf"

using namespace llvm;
using namespace seadsa;

static cl::opt<bool> TypeInfLog(
    "sea-dsa-typeinf-log",
    cl::desc("Print TypeInference assignments per function to stderr"),
    cl::init(false));

namespace {

static void printTypeName(const llvm::Type &Ty, raw_ostream &OS) {
  Ty.print(OS, /*IsForDebug=*/false, /*NoDetails=*/true);
}

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

/// Structural equality check for abstract types.
/// Hashing is avoided for now for simplicity and to avoid propagating error if uniquing fails.
/// TODO: once implementation stabilizes, consider replacing uses of this function to hashing-based equality.
static bool equalsSlow(AbsType A, AbsType B) {
  if (A == B) return true;
  if (!A || !B) return false;
  if (A->K != B->K) return false;
  switch (A->K) {
  case TypeNode::Kind::Bottom:
  case TypeNode::Kind::Top:
    return true;
  case TypeNode::Kind::Scalar:
    return A->ScalarTy == B->ScalarTy;
  case TypeNode::Kind::Ptr:
  case TypeNode::Kind::Seq:
    return equalsSlow(A->Children.front(), B->Children.front());
  case TypeNode::Kind::Prod:
  case TypeNode::Kind::Sum: {
    if (A->Children.size() != B->Children.size()) return false;
    for (size_t I = 0, E = A->Children.size(); I != E; ++I) {
      if (!equalsSlow(A->Children[I], B->Children[I])) return false;
    }
    return true;
  }
  }
  return false;
}

static void printAbsType(AbsType T, raw_ostream &OS) {
  if (!T) {
    OS << "⊤";
    return;
  }

  switch (T->K) {
  case TypeNode::Kind::Bottom:
    OS << "⊥";
    return;
  case TypeNode::Kind::Top:
    OS << "⊤";
    return;
  case TypeNode::Kind::Scalar:
    printTypeName(*T->ScalarTy, OS);
    return;
  case TypeNode::Kind::Ptr:
    OS << "Ptr(";
    printAbsType(T->Children.front(), OS);
    OS << ")";
    return;
  case TypeNode::Kind::Seq:
    OS << "Seq(";
    printAbsType(T->Children.front(), OS);
    OS << ")";
    return;
  case TypeNode::Kind::Prod: {
    OS << "(";
    for (size_t I = 0, E = T->Children.size(); I < E; ++I) {
      if (I) OS << " × ";
      printAbsType(T->Children[I], OS);
    }
    OS << ")";
    return;
  }
  case TypeNode::Kind::Sum: {
    bool First = true;
    for (AbsType Child : T->Children) {
      if (!First) OS << " + ";
      printAbsType(Child, OS);
      First = false;
    }
    if (First) OS << "⊥";
    return;
  }
  }

  OS << "⊤";
}

static void renderTypeToBuffer(AbsType T, SmallVectorImpl<char> &Buffer) {
  Buffer.clear();
  raw_svector_ostream OS(Buffer);
  printAbsType(T, OS);
}

} // namespace

//===----------------------------------------------------------------------===
// AbstractType
//===----------------------------------------------------------------------===

AbstractType::Kind AbstractType::kind() const {
  return m_type ? m_type->K : TypeNode::Kind::Top;
}

AbsType AbstractType::pointee() const {
  if (!m_type || m_type->Children.empty()) return nullptr;
  return m_type->Children.front();
}

ArrayRef<AbsType> AbstractType::elements() const {
  if (!m_type) return {};
  return m_type->Children;
}

void AbstractType::print(raw_ostream &OS) const { printAbsType(m_type, OS); }

bool AbstractType::equals(const AbstractType &Other) const {
  return equalsSlow(m_type, Other.m_type);
}

//===----------------------------------------------------------------------===
// AbstractTypeContext
//===----------------------------------------------------------------------===

struct AbstractTypeContext::Key {
  TypeNode::Kind K = TypeNode::Kind::Top;
  const llvm::Type *Scalar = nullptr;
  SmallVector<AbsType, 4> Children;
};

struct AbstractTypeContext::KeyInfo {
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

AbstractTypeContext::AbstractTypeContext() {
  Key BottomKey;
  BottomKey.K = TypeNode::Kind::Bottom;
  m_bottom = uniquedInsert(std::move(BottomKey));

  Key TopKey;
  TopKey.K = TypeNode::Kind::Top;
  m_top = uniquedInsert(std::move(TopKey));
}

AbsType AbstractTypeContext::uniquedInsert(Key K) {
  auto It = m_cache.find(K);
  if (It != m_cache.end()) return It->second.get();

  auto Node = std::make_unique<TypeNode>();
  Node->K = K.K;
  Node->ScalarTy = K.Scalar;
  Node->Children.assign(K.Children.begin(), K.Children.end());
  AbsType Result = Node.get();
  m_cache.try_emplace(std::move(K), std::move(Node));
  return Result;
}

bool AbstractTypeContext::formsCycle(AbsType Candidate,
                                     ArrayRef<AbsType> Children) const {
  // With immutable nodes and outward-only edges, cycles cannot be created.
  // This guard checks for pathological existing cycles defensively.
  SmallPtrSet<AbsType, 16> Visited;
  SmallVector<AbsType, 16> Worklist(Children.begin(), Children.end());
  while (!Worklist.empty()) {
    AbsType Cur = Worklist.pop_back_val();
    if (!Cur || !Visited.insert(Cur).second) continue;
    if (Cur == Candidate) return true;
    Worklist.append(Cur->Children.begin(), Cur->Children.end());
  }
  return false;
}

AbsType AbstractTypeContext::mkBottom() { return m_bottom; }

AbsType AbstractTypeContext::mkTop() { return m_top; }

AbsType AbstractTypeContext::mkScalar(const llvm::Type *Ty) {
  Key K;
  K.K = TypeNode::Kind::Scalar;
  K.Scalar = Ty;
  return uniquedInsert(std::move(K));
}

AbsType AbstractTypeContext::mkPtr(AbsType Pointee) {
  Key K;
  K.K = TypeNode::Kind::Ptr;
  K.Children.push_back(Pointee);
  AbsType Temp = reinterpret_cast<AbsType>(this); // placeholder identity
  if (formsCycle(Temp, K.Children)) return m_top;
  return uniquedInsert(std::move(K));
}

AbsType AbstractTypeContext::mkSeq(AbsType Element) {
  Key K;
  K.K = TypeNode::Kind::Seq;
  K.Children.push_back(Element);
  AbsType Temp = reinterpret_cast<AbsType>(this);
  if (formsCycle(Temp, K.Children)) return m_top;
  return uniquedInsert(std::move(K));
}

AbsType AbstractTypeContext::mkProd(ArrayRef<AbsType> Elements) {
  Key K;
  K.K = TypeNode::Kind::Prod;
  for (AbsType E : Elements) {
    if (E && E->K == TypeNode::Kind::Prod) {
      K.Children.append(E->Children.begin(), E->Children.end());
    } else {
      K.Children.push_back(E);
    }
  }
  AbsType Temp = reinterpret_cast<AbsType>(this);
  if (formsCycle(Temp, K.Children)) return m_top;
  return uniquedInsert(std::move(K));
}

AbsType AbstractTypeContext::mkSum(ArrayRef<AbsType> Summands) {
  SmallVector<AbsType, 8> Flat;
  for (AbsType S : Summands) {
    if (!S || S == m_bottom) continue;
    if (S->K == TypeNode::Kind::Top) return m_top;
    if (S->K == TypeNode::Kind::Sum) {
      Flat.append(S->Children.begin(), S->Children.end());
    } else {
      Flat.push_back(S);
    }
  }

  // Deduplicate and sort for determinism.
  SmallString<64> BufferA, BufferB;
  std::stable_sort(Flat.begin(), Flat.end(), [&](AbsType A, AbsType B) {
    if (A == B) return false;
    renderTypeToBuffer(A, BufferA);
    renderTypeToBuffer(B, BufferB);
    return BufferA.str() < BufferB.str();
  });
  Flat.erase(std::unique(Flat.begin(), Flat.end()), Flat.end());

  if (Flat.empty()) return m_bottom;
  if (Flat.size() == 1) return Flat.front();

  Key K;
  K.K = TypeNode::Kind::Sum;
  K.Children.append(Flat.begin(), Flat.end());
  AbsType Temp = reinterpret_cast<AbsType>(this);
  if (formsCycle(Temp, K.Children)) return m_top;
  return uniquedInsert(std::move(K));
}

AbsType AbstractTypeContext::join(AbsType A, AbsType B) {
  if (equals(A, B)) return A;
  return mkSum({A, B});
}

bool AbstractTypeContext::equals(AbsType A, AbsType B) const {
  return equalsSlow(A, B);
}

void AbstractTypeContext::print(AbsType T, raw_ostream &OS) const {
  printAbsType(T, OS);
}

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
