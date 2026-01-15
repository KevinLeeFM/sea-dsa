#include "seadsa/AbstractType.hh"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>

using namespace llvm;
using namespace seadsa;

namespace {

static void printTypeName(const llvm::Type &Ty, raw_ostream &OS) {
  Ty.print(OS, /*IsForDebug=*/false, /*NoDetails=*/true);
}

/// Structural equality (not equivalence) for AbsType nodes.
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

/// Split a flattened Prod into (left, right) where right is the suffix.
static void splitProdLeftRight(AbsType P, AbsType &Left, AbsType &Right,
                               AbstractTypeContext &Ctx) {
  assert(P && P->K == TypeNode::Kind::Prod && "splitProdLeftRight on non-prod");
  assert(!P->Children.empty());
  Left = P->Children.front();
  if (P->Children.size() == 2) {
    Right = P->Children[1];
    return;
  }
  SmallVector<AbsType, 8> Rest(P->Children.begin() + 1, P->Children.end());
  Right = Ctx.mkProd(Rest);
}

static bool isTop(AbsType T) { return T && T->K == TypeNode::Kind::Top; }
static bool isBottom(AbsType T) { return T && T->K == TypeNode::Kind::Bottom; }

} // namespace

//===----------------------------------------------------------------------===
// AbstractType
//===----------------------------------------------------------------------===

AbstractType::Kind AbstractType::kind() const {
  return m_type ? m_type->K : TypeNode::Kind::Top;
}

AbsType AbstractType::pointee() const {
  if (!m_type) return nullptr;
  if (m_type->K != TypeNode::Kind::Ptr && m_type->K != TypeNode::Kind::Seq)
    return nullptr;
  if (m_type->Children.empty()) return nullptr;
  return m_type->Children.front();
}

ArrayRef<AbsType> AbstractType::elements() const {
  if (!m_type) return {};
  if (m_type->K != TypeNode::Kind::Prod && m_type->K != TypeNode::Kind::Sum)
    return {};
  return m_type->Children;
}

void AbstractType::print(raw_ostream &OS) const { printAbsType(m_type, OS); }

bool AbstractType::equals(const AbstractType &Other) const {
  return equalsSlow(m_type, Other.m_type);
}

//===----------------------------------------------------------------------===
// AbstractTypeContext
//===----------------------------------------------------------------------===

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
  assert(Ty && "mkScalar requires non-null llvm::Type*");
  assert((Ty->isIntegerTy() || Ty->isFloatingPointTy()) &&
         "mkScalar expects an integer or floating-point LLVM type");
  Key K;
  K.K = TypeNode::Kind::Scalar;
  K.Scalar = Ty;
  return uniquedInsert(std::move(K));
}

AbsType AbstractTypeContext::mkPtr(AbsType Pointee) {
  assert(Pointee && "mkPtr requires non-null pointee type");
  AbsType N = normalize(Pointee);
  Key K;
  K.K = TypeNode::Kind::Ptr;
  K.Children.push_back(N);
  AbsType Temp = reinterpret_cast<AbsType>(this);
  if (formsCycle(Temp, K.Children)) return m_top;
  return uniquedInsert(std::move(K));
}

AbsType AbstractTypeContext::normalizeSeq(AbsType Element) {
  AbsType Elem = normalize(Element);
  if (Elem->K == TypeNode::Kind::Seq) return Elem;
  Key K;
  K.K = TypeNode::Kind::Seq;
  K.Children.push_back(Elem);
  AbsType Temp = reinterpret_cast<AbsType>(this);
  if (formsCycle(Temp, K.Children)) return m_top;
  return uniquedInsert(std::move(K));
}

AbsType AbstractTypeContext::mkSeq(AbsType Element) {
  assert(Element && "mkSeq requires non-null element type");
  return normalizeSeq(Element);
}

AbsType AbstractTypeContext::normalizeProdChildren(ArrayRef<AbsType> Elements) {
  assert(llvm::all_of(Elements, [](AbsType E) { return E; }) &&
         "normalizeProdChildren requires non-null elements");
  SmallVector<AbsType, 8> Flat;
  for (AbsType E : Elements) {
    AbsType N = normalize(E);
    if (N->K == TypeNode::Kind::Prod) {
      Flat.append(N->Children.begin(), N->Children.end());
    } else {
      Flat.push_back(N);
    }
  }

  if (Flat.size() >= 2) {
    AbsType Back = Flat.back();
    if (Back->K == TypeNode::Kind::Seq) {
      AbsType Elem = Back->Children.front();
      if (equalsSlow(Elem, Flat[Flat.size() - 2])) return Back;
    }
  }

  if (Flat.size() == 1) return Flat.front();

  Key K;
  K.K = TypeNode::Kind::Prod;
  K.Children.append(Flat.begin(), Flat.end());
  AbsType Temp = reinterpret_cast<AbsType>(this);
  if (formsCycle(Temp, K.Children)) return m_top;
  return uniquedInsert(std::move(K));
}

AbsType AbstractTypeContext::mkProd(ArrayRef<AbsType> Elements) {
  return normalizeProdChildren(Elements);
}

AbsType AbstractTypeContext::normalizeSum(ArrayRef<AbsType> Summands) {
  SmallVector<AbsType, 16> Items;
  for (AbsType S : Summands) {
    assert(S && "mkSum requires non-null summands; pass Top explicitly if needed");
    AbsType N = normalize(S);
    if (!N || isBottom(N)) continue;
    if (isTop(N)) return m_top;
    if (N->K == TypeNode::Kind::Sum) {
      for (AbsType Child : N->Children) {
        if (llvm::is_contained(Items, Child)) continue;
        Items.push_back(Child);
      }
    } else {
      if (llvm::is_contained(Items, N)) continue;
      Items.push_back(N);
    }
  }

  // Distribute products sharing the same right component: (a×r) + (b×r) = (a+b)×r
  SmallVector<AbsType, 16> Others;
  DenseMap<AbsType, SmallVector<AbsType, 4>> Groups;
  for (AbsType N : Items) {
    if (N->K == TypeNode::Kind::Prod && !N->Children.empty()) {
      AbsType L = nullptr, R = nullptr;
      splitProdLeftRight(N, L, R, *this);
      Groups[R].push_back(L);
    } else {
      Others.push_back(N);
    }
  }

  SmallVector<AbsType, 16> Distributed(Others.begin(), Others.end());
  if (!Groups.empty()) {
    SmallVector<AbsType, 8> Keys;
    for (auto &KV : Groups) {
      Keys.push_back(KV.first);
    }
    SmallString<64> BufferA, BufferB;
    std::stable_sort(Keys.begin(), Keys.end(), [&](AbsType A, AbsType B) {
      if (A == B) return false;
      renderTypeToBuffer(A, BufferA);
      renderTypeToBuffer(B, BufferB);
      return BufferA.str() < BufferB.str();
    });
    Keys.erase(std::unique(Keys.begin(), Keys.end()), Keys.end());

    for (AbsType Key : Keys) {
      auto It = Groups.find(Key);
      if (It == Groups.end()) continue;
      auto &Lefts = It->second;
      AbsType Combined = Lefts.size() == 1 ? Lefts.front() : mkSum(Lefts);
      Distributed.push_back(mkProd({Combined, Key}));
    }
  }

  if (Distributed.empty()) return m_bottom;
  if (Distributed.size() == 1) return Distributed.front();

  SmallString<64> BufferA, BufferB;
  std::stable_sort(Distributed.begin(), Distributed.end(), [&](AbsType A, AbsType B) {
    if (A == B) return false;
    renderTypeToBuffer(A, BufferA);
    renderTypeToBuffer(B, BufferB);
    return BufferA.str() < BufferB.str();
  });
  Distributed.erase(std::unique(Distributed.begin(), Distributed.end()), Distributed.end());

  Key K;
  K.K = TypeNode::Kind::Sum;
  K.Children.append(Distributed.begin(), Distributed.end());
  AbsType Temp = reinterpret_cast<AbsType>(this);
  if (formsCycle(Temp, K.Children)) return m_top;
  return uniquedInsert(std::move(K));
}

AbsType AbstractTypeContext::mkSum(ArrayRef<AbsType> Summands) {
  return normalizeSum(Summands);
}

AbsType AbstractTypeContext::join(AbsType A, AbsType B) {
  if (equivalent(A, B)) return normalize(A);
  return mkSum({A, B});
}

bool AbstractTypeContext::equals(AbsType A, AbsType B) const {
  return equalsSlow(A, B);
}

void AbstractTypeContext::print(AbsType T, raw_ostream &OS) const {
  printAbsType(T, OS);
}

AbsType AbstractTypeContext::normalize(AbsType T) {
  if (!T) return mkTop();
  switch (T->K) {
  case TypeNode::Kind::Bottom:
    return mkBottom();
  case TypeNode::Kind::Top:
    return mkTop();
  case TypeNode::Kind::Scalar:
    return mkScalar(T->ScalarTy);
  case TypeNode::Kind::Ptr:
    return mkPtr(normalize(T->Children.front()));
  case TypeNode::Kind::Seq:
    return normalizeSeq(T->Children.front());
  case TypeNode::Kind::Prod:
    return normalizeProdChildren(T->Children);
  case TypeNode::Kind::Sum:
    return normalizeSum(T->Children);
  }
  return mkTop();
}

bool AbstractTypeContext::equivalent(AbsType A, AbsType B) {
  return normalize(A) == normalize(B);
}

static bool leqImpl(AbsType L, AbsType R, AbstractTypeContext &Ctx);

AbsType AbstractTypeContext::deref(AbsType T) {
  AbsType N = normalize(T);
  if (isBottom(N)) return m_bottom;
  if (N->K == TypeNode::Kind::Ptr) return N->Children.front();
  if (N->K == TypeNode::Kind::Sum) {
    SmallVector<AbsType, 8> Parts;
    for (AbsType S : N->Children) {
      Parts.push_back(deref(S));
    }
    return mkSum(Parts);
  }
  return m_bottom;
}

AbsType AbstractTypeContext::trunc(AbsType T, unsigned Index) {
  if (Index == 0) return normalize(T);
  AbsType N = normalize(T);
  if (isTop(N)) return m_top;
  if (isBottom(N)) return m_bottom;
  if (N->K == TypeNode::Kind::Seq) return N;
  if (N->K == TypeNode::Kind::Prod) {
    AbsType L = nullptr, R = nullptr;
    splitProdLeftRight(N, L, R, *this);
    return trunc(R, Index - 1);
  }
  if (N->K == TypeNode::Kind::Sum) {
    SmallVector<AbsType, 8> Truncated;
    unsigned Arity = 0;
    for (AbsType S : N->Children) {
      if (S->K != TypeNode::Kind::Prod) return m_bottom;
      SmallVector<AbsType, 8> Parts;
      for (AbsType P : S->Children) {
        Parts.push_back(P);
      }
      if (Arity == 0) Arity = Parts.size();
      if (Parts.size() != Arity) return m_bottom;
      if (Index >= Parts.size()) return m_bottom;
      SmallVector<AbsType, 8> Suffix(Parts.begin() + Index, Parts.end());
      AbsType Tail = Suffix.size() == 1 ? Suffix.front() : mkProd(Suffix);
      Truncated.push_back(Tail);
    }
    return mkSum(Truncated);
  }
  return m_bottom;
}

bool AbstractTypeContext::leq(AbsType A, AbsType B) {
  return leqImpl(normalize(A), normalize(B), *this);
}

static bool leqImpl(AbsType L, AbsType R, AbstractTypeContext &Ctx) {
  if (L == R) return true;
  if (Ctx.equivalent(L, R)) return true;
  if (isBottom(L)) return true;
  if (isTop(R)) return true;

  if (L && L->K == TypeNode::Kind::Sum) {
    for (AbsType S : L->Children) {
      if (!leqImpl(S, R, Ctx)) return false;
    }
    return true;
  }

  if (R && R->K == TypeNode::Kind::Sum) {
    for (AbsType S : R->Children) {
      if (leqImpl(L, S, Ctx)) return true;
    }
    return false;
  }

  if (L && R && L->K == TypeNode::Kind::Ptr && R->K == TypeNode::Kind::Ptr) {
    return leqImpl(L->Children.front(), R->Children.front(), Ctx);
  }

  if (L && R && L->K == TypeNode::Kind::Prod && R->K == TypeNode::Kind::Prod) {
    if (L->Children.size() != R->Children.size()) return false;
    for (size_t I = 0, E = L->Children.size(); I != E; ++I) {
      if (!leqImpl(L->Children[I], R->Children[I], Ctx)) return false;
    }
    return true;
  }

  if (L && R && L->K == TypeNode::Kind::Prod && R->K == TypeNode::Kind::Seq) {
    AbsType LL = nullptr, LR = nullptr;
    splitProdLeftRight(L, LL, LR, Ctx);
    if (Ctx.equivalent(R->Children.front(), LL) &&
        leqImpl(LR, Ctx.mkSeq(LL), Ctx))
      return true;
  }

  if (L && R && L->K == TypeNode::Kind::Seq && R->K == TypeNode::Kind::Prod) {
    AbsType RL = nullptr, RR = nullptr;
    splitProdLeftRight(R, RL, RR, Ctx);
    if (isTop(RR)) return leqImpl(L->Children.front(), RL, Ctx);
    if (RR->K == TypeNode::Kind::Seq && Ctx.equivalent(RL, RR->Children.front()))
      return Ctx.equivalent(L->Children.front(), RR->Children.front());
  }

  if (L && R && L->K == TypeNode::Kind::Seq && R->K == TypeNode::Kind::Seq) {
    return leqImpl(L->Children.front(), R->Children.front(), Ctx);
  }

  if (R && R->K == TypeNode::Kind::Seq) {
    if (leqImpl(L, R->Children.front(), Ctx)) return true;
    if (L && L->K == TypeNode::Kind::Prod) {
      AbsType LL = nullptr, LR = nullptr;
      splitProdLeftRight(L, LL, LR, Ctx);
      if (leqImpl(LL, R->Children.front(), Ctx) && leqImpl(LR, R, Ctx))
        return true;
    }
    if (L && L->K == TypeNode::Kind::Seq &&
        Ctx.equivalent(L->Children.front(), R->Children.front()))
      return true;
  }

  return false;
}

AbsType AbstractTypeContext::meetTwo(AbsType A, AbsType B) {
  if (leq(A, B)) return normalize(A);
  if (leq(B, A)) return normalize(B);

  AbsType NA = normalize(A);
  AbsType NB = normalize(B);

  if (NA->K == TypeNode::Kind::Sum && NB->K == TypeNode::Kind::Sum) {
    SmallVector<AbsType, 16> Candidates;
    for (AbsType SA : NA->Children) {
      for (AbsType SB : NB->Children) {
        AbsType M = meetTwo(SA, SB);
        if (!isBottom(M)) Candidates.push_back(M);
      }
    }
    return mkSum(Candidates);
  }

  auto meetSumOther = [&](AbsType SumTy, AbsType Other) {
    SmallVector<AbsType, 16> Cands;
    for (AbsType S : SumTy->Children) {
      AbsType M = meetTwo(S, Other);
      if (!isBottom(M)) Cands.push_back(M);
    }
    return mkSum(Cands);
  };

  if (NA->K == TypeNode::Kind::Sum) return meetSumOther(NA, NB);
  if (NB->K == TypeNode::Kind::Sum) return meetSumOther(NB, NA);

  if (NA->K == TypeNode::Kind::Prod && NB->K == TypeNode::Kind::Prod) {
    if (NA->Children.size() != NB->Children.size()) return m_bottom;
    SmallVector<AbsType, 8> Parts;
    for (size_t I = 0, E = NA->Children.size(); I != E; ++I) {
      AbsType M = meetTwo(NA->Children[I], NB->Children[I]);
      if (isBottom(M)) return m_bottom;
      Parts.push_back(M);
    }
    return mkProd(Parts);
  }

  if (NA->K == TypeNode::Kind::Ptr && NB->K == TypeNode::Kind::Ptr) {
    AbsType P = meetTwo(NA->Children.front(), NB->Children.front());
    if (isBottom(P)) return m_bottom;
    return mkPtr(P);
  }

  if (NA->K == TypeNode::Kind::Seq && NB->K == TypeNode::Kind::Seq) {
    AbsType E = meetTwo(NA->Children.front(), NB->Children.front());
    if (isBottom(E)) return m_bottom;
    return mkSeq(E);
  }

  return m_bottom;
}

AbsType AbstractTypeContext::meet(ArrayRef<AbsType> Types) {
  if (Types.empty()) return mkTop();
  AbsType Result = normalize(Types.front());
  for (size_t I = 1, E = Types.size(); I < E; ++I) {
    Result = meetTwo(Result, Types[I]);
  }
  return Result;
}
