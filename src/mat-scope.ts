// mat-scope.ts — RAII-style scope for OpenCV Mat lifecycle management.
//
// Tracks CvMat (and CvMatVector) allocations and bulk-deletes them when
// the scope exits.  Replaces the ad-hoc toDelete/mat() pattern.

interface Deletable {
  delete(): void;
}

export class MatScope {
  private tracked: Deletable[] = [];

  /** Track a Mat (or MatVector) for deletion when this scope exits. */
  track<T extends Deletable>(m: T): T {
    this.tracked.push(m);
    return m;
  }

  /** Remove a Mat from this scope (e.g. to transfer ownership to the caller). */
  untrack(m: Deletable): void {
    const i = this.tracked.indexOf(m);
    if (i >= 0) this.tracked.splice(i, 1);
  }

  /** Delete all tracked Mats. Safe to call multiple times. */
  release(): void {
    for (const m of this.tracked) {
      try { m.delete(); } catch { /* already deleted */ }
    }
    this.tracked.length = 0;
  }

  /** Run a function with this scope, releasing all tracked Mats on exit. */
  run<T>(fn: (scope: MatScope) => T): T {
    try {
      return fn(this);
    } finally {
      this.release();
    }
  }
}
