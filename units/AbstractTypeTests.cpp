#include "doctest.h"

#include "seadsa/AbstractType.hh"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"

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

static AbstractTypeContext makeCtx() { return AbstractTypeContext(); }

// Build expected Ptr(Top) once per context.
static AbsType ptrTop(AbstractTypeContext &C) { return C.mkPtr(C.mkTop()); }

//===----------------------------------------------------------------------===
// Builder happy-path tests
//===----------------------------------------------------------------------===

TEST_CASE("AbstractType.Builders.BasicKinds") {
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

TEST_CASE("AbstractType.Builders.SeqProdSum") {
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

TEST_CASE("AbstractType.Builders.SumDedupAndFlatten") {
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

TEST_CASE("AbstractType.Builders.SumProperties") {
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

TEST_CASE("AbstractType.Builders.ProdProperties") {
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

TEST_CASE("AbstractType.Builders.SingleElementProdCanonicalizes") {
  auto C = makeCtx();
  AbsType a = C.mkScalar(Type::getInt32Ty(ctx()));
  AbsType p = C.mkProd({a});
  CHECK(p == a);
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
// Lattice ordering / equivalence tests (ported from docs/tests/test_types.py)
//===----------------------------------------------------------------------===

TEST_CASE("Lattice.EquivalentTrailingTop") {
  auto C = makeCtx();
  AbsType tau = C.mkProd({C.mkScalar(Type::getInt32Ty(ctx())), C.mkTop()});
  AbsType i32 = C.mkScalar(Type::getInt32Ty(ctx()));
  CHECK_FALSE(C.equivalent(tau, i32));
  CHECK_FALSE(C.leq(tau, i32));
  CHECK_FALSE(C.leq(i32, tau));
}

TEST_CASE("Lattice.SeqFlattening") {
  auto C = makeCtx();
  AbsType nested = C.mkSeq(C.mkSeq(C.mkScalar(Type::getInt32Ty(ctx()))));
  AbsType flat = C.mkSeq(C.mkScalar(Type::getInt32Ty(ctx())));
  CHECK(C.equivalent(nested, flat));
}

TEST_CASE("Lattice.TruncProductSuffix") {
  auto C = makeCtx();
  AbsType i = C.mkScalar(Type::getInt32Ty(ctx()));
  AbsType ptrI = C.mkPtr(i);
  AbsType prod = C.mkProd({i, ptrI, i});

  AbsType t1 = C.trunc(prod, 1);
  AbsType t2 = C.trunc(prod, 2);
  AbsType t3 = C.trunc(prod, 3);

  CHECK(C.equivalent(t1, C.mkProd({ptrI, i})));
  CHECK(C.equivalent(t2, i));
  CHECK(t3->K == TypeNode::Kind::Bottom);
}

TEST_CASE("Lattice.TruncUnionRequiresMatchingProducts") {
  auto C = makeCtx();
  AbsType i = C.mkScalar(Type::getInt32Ty(ctx()));
  AbsType ptrI = C.mkPtr(i);
  AbsType ptrTop = C.mkPtr(C.mkTop());

  AbsType prodA = C.mkProd({i, ptrI});
  AbsType prodB = C.mkProd({i, ptrTop});
  AbsType uni = C.mkSum({prodA, prodB});

  AbsType truncated = C.trunc(uni, 1);
  AbsType expected = C.mkSum({ptrI, ptrTop});
  CHECK(C.leq(truncated, expected));
}

TEST_CASE("Lattice.TruncUnionMismatchFallsBackToBottom") {
  auto C = makeCtx();
  AbsType i = C.mkScalar(Type::getInt32Ty(ctx()));
  AbsType prod = C.mkProd({i, i});
  AbsType uni = C.mkSum({i, prod});

  AbsType truncated = C.trunc(uni, 1);
  CHECK(truncated->K == TypeNode::Kind::Bottom);
}

TEST_CASE("Lattice.DerefDistributesOverUnions") {
  auto C = makeCtx();
  AbsType i = C.mkScalar(Type::getInt32Ty(ctx()));
  AbsType ptrI = C.mkPtr(i);
  AbsType ptrTop = C.mkPtr(C.mkTop());
  AbsType uni = C.mkSum({ptrI, ptrTop});

  AbsType d = C.deref(uni);
  AbsType expected = C.mkSum({i, C.mkTop()});
  CHECK(C.equivalent(d, expected));
}

TEST_CASE("Lattice.PointerCovariance") {
  auto C = makeCtx();
  AbsType i = C.mkScalar(Type::getInt32Ty(ctx()));
  AbsType ptrI = C.mkPtr(i);
  AbsType ptrTop = C.mkPtr(C.mkTop());

  CHECK(C.leq(ptrI, ptrTop));
  CHECK_FALSE(C.leq(ptrTop, ptrI));
}

TEST_CASE("Lattice.MeetRespectsStructures") {
  auto C = makeCtx();
  AbsType i = C.mkScalar(Type::getInt32Ty(ctx()));
  AbsType ptrI = C.mkPtr(i);
  AbsType ptrTop = C.mkPtr(C.mkTop());

  AbsType left = C.mkPtr(C.mkSum({i, ptrI}));
  AbsType right = C.mkPtr(C.mkSum({i, ptrTop}));
  AbsType meet = C.meetTwo(left, right);
  bool same = (meet == left);
  CHECK(same);
}

TEST_CASE("Lattice.SeqIntroAndProdRules") {
  auto C = makeCtx();
  AbsType i = C.mkScalar(Type::getInt32Ty(ctx()));
  AbsType seqI = C.mkSeq(i);
  AbsType prodSeq = C.mkProd({i, seqI});

  CHECK(C.leq(i, seqI));
  CHECK(C.leq(prodSeq, seqI));
  CHECK(C.equivalent(prodSeq, seqI));
}

TEST_CASE("Lattice.TruncRespectsTopBotAndSeq") {
  auto C = makeCtx();
  AbsType i = C.mkScalar(Type::getInt32Ty(ctx()));
  AbsType seq = C.mkSeq(i);

  bool topSame = (C.trunc(C.mkTop(), 2) == C.mkTop());
  bool botSame = (C.trunc(C.mkBottom(), 5) == C.mkBottom());
  bool seqSame = (C.trunc(seq, 3) == seq);
  CHECK(topSame);
  CHECK(botSame);
  CHECK(seqSame);
}

TEST_CASE("Lattice.SumOfProductsDistribution") {
  auto C = makeCtx();
  AbsType i = C.mkScalar(Type::getInt32Ty(ctx()));
  AbsType ptrI = C.mkPtr(i);

  AbsType leftSum = C.mkSum({i, ptrI});
  AbsType distributed = C.mkSum({C.mkProd({i, i}), C.mkProd({ptrI, i})});
  AbsType combined = C.mkProd({leftSum, i});

  CHECK(C.equivalent(distributed, combined));
}

TEST_CASE("Utility.MayContainPointer.Positive") {
  auto C = makeCtx();
  AbsType i32 = C.mkScalar(Type::getInt32Ty(ctx()));
  AbsType ptrI = C.mkPtr(i32);

  CHECK(C.mayContainPointer(ptrI));
  CHECK(C.mayContainPointer(C.mkProd({i32, C.mkTop()})));
  CHECK(C.mayContainPointer(C.mkSum({ptrI, i32})));
  CHECK(C.mayContainPointer(C.mkProd({ptrI, i32})));
  CHECK(C.mayContainPointer(C.mkSeq(ptrI)));
}

TEST_CASE("Utility.MayContainPointer.Negative") {
  auto C = makeCtx();
  AbsType i32 = C.mkScalar(Type::getInt32Ty(ctx()));
  AbsType i64 = C.mkScalar(Type::getInt64Ty(ctx()));

  CHECK_FALSE(C.mayContainPointer(i32));
  CHECK_FALSE(C.mayContainPointer(C.mkBottom()));
  CHECK_FALSE(C.mayContainPointer(C.mkProd({i32, i64})));
  CHECK_FALSE(C.mayContainPointer(C.mkSeq(i32)));
  CHECK_FALSE(C.mayContainPointer(C.mkSum({i32, C.mkBottom()})));
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
