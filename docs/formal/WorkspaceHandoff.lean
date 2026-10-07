-- Abstract file-store model. Filesystem execution is tested separately.
-- Keys represent caller-owned, unique candidate paths; no GPU claim is made.
namespace WorkspaceHandoff
structure Artifact where
  source : String
  evidence : String
  deriving DecidableEq
abbrev Store := Nat → Option Artifact

def put (s : Store) (key : Nat) (a : Artifact) : Option Store :=
  match s key with
  | none => some (fun k => if k = key then some a else s k)
  | some old => if old = a then some s else none

theorem collision_refused (s : Store) (k : Nat) (old a : Artifact)
    (existing : s k = some old) (different : old ≠ a) : put s k a = none := by
  simp [put, existing, different]

theorem same_content_idempotent (s : Store) (k : Nat) (a : Artifact)
    (existing : s k = some a) : put s k a = some s := by
  simp [put, existing]

theorem fresh_source_preserved (s : Store) (k : Nat) (a : Artifact)
    (fresh : s k = none) :
    put s k a = some (fun j => if j = k then some a else s j) := by
  simp [put, fresh]

theorem fresh_other_candidate_unchanged (s : Store) (k j : Nat) (a : Artifact)
    (different : j ≠ k) :
    (fun x => if x = k then some a else s x) j = s j := by
  simp [different]

structure Record where
  key : Nat
  raw : Artifact
structure Index where
  key : Nat
  deriving DecidableEq

def project (r : Record) : Index := ⟨r.key⟩
theorem reference_identity (r : Record) : (project r).key = r.key := rfl

theorem projection_preserves_access (s : Store) (r : Record)
    (bound : s r.key = some r.raw) : s (project r).key = some r.raw := by
  exact bound

theorem projection_size_independent_of_source (k : Nat) (a b : Artifact) :
    project ⟨k,a⟩ = project ⟨k,b⟩ := rfl

theorem distinct_candidates_not_aliased (a b : Record) (distinct : a.key ≠ b.key) :
    project a ≠ project b := by
  intro equal
  apply distinct
  exact congrArg Index.key equal
end WorkspaceHandoff
