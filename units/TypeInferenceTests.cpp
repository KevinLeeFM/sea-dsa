#include "doctest.h"

#include "seadsa/TypeInference.hh"

#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include <algorithm>

using namespace llvm;
using namespace seadsa;

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
  CHECK(t->K == TypeNode::Kind::Scalar);
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
