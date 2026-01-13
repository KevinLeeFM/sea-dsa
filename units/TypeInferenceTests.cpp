#include "doctest.h"

#include "seadsa/TypeInference.hh"

#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <algorithm>
#include <string>

using namespace llvm;
using namespace seadsa;

// Provided by the custom main to re-exec for death checks.
int seadsa_runSubprocessExpectFailure(const char *flagName);

//===----------------------------------------------------------------------===
// Helpers
//===----------------------------------------------------------------------===

static LLVMContext &ctx() {
  static LLVMContext C;
  return C;
}

static DataLayout dl64() {
  // Deterministic x86_64 layout.
  return DataLayout("e-m:e-p:64:64-i64:64-f80:128-n8:16:32:64-S128");
}

static DataLayout dl32() {
  // Deterministic 32-bit layout.
  return DataLayout("e-p:32:32-i64:64-n8:16:32-S128");
}

static AbstractTypeContext makeCtx() { return AbstractTypeContext(); }

// Build expected Ptr(Top) once per context.
static AbsType ptrTop(AbstractTypeContext &C) { return C.mkPtr(C.mkTop()); }

//===----------------------------------------------------------------------===
// Builder happy-path tests
//===----------------------------------------------------------------------===

TEST_CASE("TypeInference.Builders.BasicKinds") {
  auto C = makeCtx();
  AbsType bot = C.mkBottom();
  AbsType top = C.mkTop();
  CHECK(bot->K == TypeNode::Kind::Bottom);
  CHECK(top->K == TypeNode::Kind::Top);

  AbsType i32 = C.mkScalar(Type::getInt32Ty(ctx()));
  CHECK(i32->K == TypeNode::Kind::Scalar);
  CHECK(!!i32->ScalarTy);
  CHECK(i32->ScalarTy->isIntegerTy(32));

  AbsType pTop = ptrTop(C);
  CHECK(pTop->K == TypeNode::Kind::Ptr);
  bool childIsTop = (pTop->Children.front() == C.mkTop());
  CHECK(childIsTop);
}

TEST_CASE("TypeInference.Builders.SeqProdSum") {
  auto C = makeCtx();
  AbsType i32 = C.mkScalar(Type::getInt32Ty(ctx()));
  AbsType seq = C.mkSeq(i32);
  CHECK(seq->K == TypeNode::Kind::Seq);
  bool seqElemIsI32 = (seq->Children.front() == i32);
  CHECK(seqElemIsI32);

  AbsType prod = C.mkProd({i32, C.mkTop()});
  CHECK(prod->K == TypeNode::Kind::Prod);
  CHECK(prod->Children.size() == 2);

  AbsType sum = C.mkSum({i32, C.mkTop()});
  bool sumIsTop = (sum == C.mkTop());
  CHECK(sumIsTop);
}

TEST_CASE("TypeInference.Builders.SumDedupAndFlatten") {
  auto C = makeCtx();
  AbsType i64 = C.mkScalar(Type::getInt64Ty(ctx()));
  AbsType pTop = ptrTop(C);
  AbsType nested = C.mkSum({i64, C.mkSum({pTop, i64}), C.mkBottom()});
  // Should flatten, drop bottom, dedup, and keep deterministic order.
  CHECK(nested->K == TypeNode::Kind::Sum);
  CHECK(nested->Children.size() == 2);
  bool hasFirst = nested->Children[0] == i64 || nested->Children[0] == pTop;
  bool hasSecond = nested->Children[1] == i64 || nested->Children[1] == pTop;
  CHECK(hasFirst);
  CHECK(hasSecond);
}

TEST_CASE("TypeInference.Builders.SumProperties") {
  auto C = makeCtx();
  AbsType a = C.mkScalar(Type::getInt32Ty(ctx()));
  AbsType b = C.mkScalar(Type::getInt64Ty(ctx()));
  AbsType ab = C.mkSum({a, b});
  AbsType ba = C.mkSum({b, a});
  bool commutes = (ab == ba);
  CHECK(commutes);

  AbsType dup = C.mkSum({a, a});
  bool idempotent = (dup == a);
  CHECK(idempotent);

  AbsType nested = C.mkSum({ab, C.mkBottom()});
  bool flattens = (nested == ab);
  CHECK(flattens);

  // Deterministic print
  std::string S1, S2;
  raw_string_ostream OS1(S1), OS2(S2);
  C.print(ab, OS1);
  C.print(ba, OS2);
  CHECK(S1 == S2);
}

TEST_CASE("TypeInference.Builders.ProdProperties") {
  auto C = makeCtx();
  AbsType a = C.mkScalar(Type::getInt32Ty(ctx()));
  AbsType b = C.mkScalar(Type::getInt64Ty(ctx()));
  AbsType p1 = C.mkProd({a, b});
  AbsType p2 = C.mkProd({p1, C.mkScalar(Type::getInt8Ty(ctx()))});
  CHECK(p1->Children.size() == 2);
  CHECK(p2->Children.size() == 3); // flattened nested prod
  bool firstIsA = (p1->Children[0] == a);
  bool secondIsB = (p1->Children[1] == b);
  CHECK(firstIsA);
  CHECK(secondIsB);
}

//===----------------------------------------------------------------------===
// Guard / death tests (debug-only)
//===----------------------------------------------------------------------===

static void requireDeath(const char *flag) {
#ifdef NDEBUG
  WARN_MESSAGE(true, "Assertions disabled (NDEBUG); death test skipped");
#else
  int code = seadsa_runSubprocessExpectFailure(flag);
  CHECK(code != 0);
#endif
}

TEST_CASE("Guards.mkScalar.null") { requireDeath("mkScalar_null"); }
TEST_CASE("Guards.mkScalar.struct") { requireDeath("mkScalar_struct"); }
TEST_CASE("Guards.mkPtr.null") { requireDeath("mkPtr_null"); }
TEST_CASE("Guards.mkSeq.null") { requireDeath("mkSeq_null"); }
TEST_CASE("Guards.mkProd.null") { requireDeath("mkProd_null"); }
TEST_CASE("Guards.mkSum.null") { requireDeath("mkSum_null"); }

//===----------------------------------------------------------------------===
// convertLLVMType tests
//===----------------------------------------------------------------------===

static AbsType conv(const Type &Ty, const DataLayout &DL,
                    AbstractTypeContext &C) {
  return convertLLVMType(Ty, DL, C);
}

TEST_CASE("Convert.SimplePointerAndInts") {
  auto C = makeCtx();
  DataLayout DL = dl64();
  Type *i32 = Type::getInt32Ty(ctx());
  Type *i64 = Type::getInt64Ty(ctx());
  Type *ptr = PointerType::get(i32, 0);

  AbsType tPtr = conv(*ptr, DL, C);
  AbsType tI32 = conv(*i32, DL, C);
  AbsType tI64 = conv(*i64, DL, C);

  CHECK(tPtr->K == TypeNode::Kind::Sum);
  CHECK(tI64->K == TypeNode::Kind::Sum); // pointer-sized int joins Ptr(Top)
  CHECK(tI32->K == TypeNode::Kind::Scalar);

  // Sum should contain Ptr(Top).
  AbsType pTop = ptrTop(C);
  bool ptrHasTop =
      std::find(tPtr->Children.begin(), tPtr->Children.end(), pTop) !=
      tPtr->Children.end();
  bool i64HasTop =
      std::find(tI64->Children.begin(), tI64->Children.end(), pTop) !=
      tI64->Children.end();
  CHECK(ptrHasTop);
  CHECK(i64HasTop);
}

TEST_CASE("Convert.PointerInterchangeable32bit") {
  auto C = makeCtx();
  DataLayout DL = dl32();
  Type *i32 = Type::getInt32Ty(ctx());
  Type *i64 = Type::getInt64Ty(ctx());

  AbsType tI32 = conv(*i32, DL, C);
  AbsType tI64 = conv(*i64, DL, C);
  AbsType pTop = ptrTop(C);

  CHECK(tI32->K == TypeNode::Kind::Sum); // pointer-sized int
  CHECK(tI64->K == TypeNode::Kind::Scalar); // not pointer-sized here

  bool i32HasTop =
      std::find(tI32->Children.begin(), tI32->Children.end(), pTop) !=
      tI32->Children.end();
  CHECK(i32HasTop);
}

TEST_CASE("Convert.StructSimple") {
  auto C = makeCtx();
  DataLayout DL = dl64();
  LLVMContext &Ctxt = ctx();
  StructType *S = StructType::create(Ctxt, "S");
  Type *Elems[] = {Type::getInt32Ty(Ctxt)};
  S->setBody(Elems);

  AbsType t = conv(*S, DL, C);
  CHECK(t->K == TypeNode::Kind::Prod);
  CHECK(t->Children.size() == 1);
  CHECK(t->Children.front()->K == TypeNode::Kind::Scalar);
}

TEST_CASE("Convert.StructNested") {
  auto C = makeCtx();
  DataLayout DL = dl64();
  LLVMContext &Ctxt = ctx();
  StructType *Inner = StructType::create(Ctxt, "Inner");
  Inner->setBody({Type::getInt64Ty(Ctxt), Type::getInt32Ty(Ctxt)});
  StructType *Outer = StructType::create(Ctxt, "Outer");
  Outer->setBody({Inner, Type::getInt8Ty(Ctxt)});

  AbsType t = conv(*Outer, DL, C);
  CHECK(t->K == TypeNode::Kind::Prod);
  CHECK(t->Children.size() == 3); // flattened inner struct

  // Expect order: (Sum(i64, Ptr(Top))), Scalar(i32), Scalar(i8)
  AbsType child0 = t->Children[0];
  AbsType child1 = t->Children[1];
  AbsType child2 = t->Children[2];

  CHECK(child0->K == TypeNode::Kind::Sum);
  CHECK(child1->K == TypeNode::Kind::Scalar);
  CHECK(child2->K == TypeNode::Kind::Scalar);
}

TEST_CASE("Convert.OpaqueStructAndAggregates") {
  auto C = makeCtx();
  DataLayout DL = dl64();
  LLVMContext &Ctxt = ctx();
  StructType *Opaque = StructType::create(Ctxt, "Opaque");
  ArrayType *Arr = ArrayType::get(Type::getInt32Ty(Ctxt), 4);
  VectorType *Vec = FixedVectorType::get(Type::getInt8Ty(Ctxt), 8);

  bool opaqueIsTop = (conv(*Opaque, DL, C) == C.mkTop());
  bool arrIsTop = (conv(*Arr, DL, C) == C.mkTop());
  bool vecIsTop = (conv(*Vec, DL, C) == C.mkTop());
  CHECK(opaqueIsTop);
  CHECK(arrIsTop);
  CHECK(vecIsTop);
}

TEST_CASE("Print.Stability") {
  auto C = makeCtx();
  AbsType a = C.mkScalar(Type::getInt32Ty(ctx()));
  AbsType b = C.mkPtr(C.mkTop());
  AbsType sum = C.mkSum({b, a});

  std::string S1, S2;
  raw_string_ostream OS1(S1), OS2(S2);
  C.print(sum, OS1);
  C.print(C.mkSum({a, b}), OS2); // reversed order should stringify identically
  CHECK(S1 == S2);
}

//===----------------------------------------------------------------------===
// Death scenario dispatcher
//===----------------------------------------------------------------------===

bool seadsa_maybeRunDeathScenario(const char *flagName) {
  std::string F(flagName ? flagName : "");
#ifdef NDEBUG
  (void)F;
  return false;
#else
  auto C = makeCtx();
  LLVMContext &Ctxt = ctx();

  if (F == "mkScalar_null") {
    C.mkScalar(nullptr);
    return true;
  }
  if (F == "mkScalar_struct") {
    StructType *S = StructType::create(Ctxt, "BadStruct");
    C.mkScalar(S);
    return true;
  }
  if (F == "mkPtr_null") {
    C.mkPtr(nullptr);
    return true;
  }
  if (F == "mkSeq_null") {
    C.mkSeq(nullptr);
    return true;
  }
  if (F == "mkProd_null") {
    AbsType elems[2] = {C.mkScalar(Type::getInt1Ty(Ctxt)), nullptr};
    C.mkProd(elems);
    return true;
  }
  if (F == "mkSum_null") {
    AbsType elems[2] = {C.mkScalar(Type::getInt1Ty(Ctxt)), nullptr};
    C.mkSum(elems);
    return true;
  }
  return false;
#endif
}
