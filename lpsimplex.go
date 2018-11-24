package lpsimplex

import (
	"fmt"
	"math"
	"os"
)

/*
Functions
---------
    LPSimplex
    LPSimplexVerboseCallback
    LPSimplexTerseCallback

*/
// OptResult carries the result of the simplex operation.
type OptResult struct {
	X       []float64
	Fun     float64
	Nitr    int
	Status  int
	Slack   []float64
	Message string
	Success bool
}

type NB_CMD int

const (
	NB_CMD_NOP   NB_CMD = 0x0 // New Behavior Do Nothing
	NB_CMD_RESET NB_CMD = 0x1 // tbd

)

var (
	lastObjectiveValue   float64 = 0.0
	degeneritePivotCount int     = 0
	dynamicBlandRule     bool    = false
)

// may want to get rid of this first parameter and alway reset
// whenever func LPSimplex() is called
// Now resetting as needed not needed here!!!
func LPSimplexSetNewBehavior(cmd NB_CMD, dynamicBland bool) int {
	dynamicBlandRule = dynamicBland
	if cmd == 0 {
		fmt.Printf("LPSimplexSetNewBehavior() called with NB_CMD_NOP\n")
	}
	if cmd&NB_CMD_RESET > 0 {
		//degeneritePivotCount = 0
		//lastObjectiveValue = 0.0
		fmt.Printf("LPSimplexSetNewBehavior() called with NB_CMD_RESET\n")
	}
	return degeneritePivotCount
}

// Callbackfunc is the function type for the callback that LPSimplex takes.
type Callbackfunc func([]float64, [][]float64, int, int, int, int, []int, bool)

// TersPrintIntArray prints, skipping middle elements, so the array fits on a line.
func TersPrintIntArray(a []int) {
	if a == nil || len(a) == 0 {
		return
	}
	l := len(a)
	if l > 8 {
		fmt.Printf(" [ %d\t%d\t%d,,,\t%d\t%d\t%d\t]\n", a[0], a[1], a[2], a[l-3], a[l-2], a[l-1])
	} else {
		fmt.Printf(" [ ")
		for i := 0; i < l; i++ {
			fmt.Printf("%d\t", a[i])
		}
		fmt.Printf("]\n")
	}
}

// TersPrintArray prints, skipping middle elements, so the array fits on a line.
func TersPrintArray(a []float64) {
	if a == nil || len(a) == 0 {
		return
	}
	l := len(a)
	if l > 8 {
		fmt.Printf(" [ %f\t%f\t%f,,,\t%f\t%f\t%f\t]\n", a[0], a[1], a[2], a[l-3], a[l-2], a[l-1])
	} else {
		fmt.Printf(" [ ")
		for i := 0; i < l; i++ {
			fmt.Printf("%f\t", a[i])
		}
		fmt.Printf("]\n")
	}
}

// TersPrintMatrix prints, skipping middle elements, so it fits on a screen.
func TersPrintMatrix(a [][]float64) error {
	r := len(a)
	c := len(a[0])
	fmt.Printf("Rows: %d, Cols: %d\n", len(a), len(a[0]))
	fmt.Printf("[\n")
	if r > 8 {
		for i := 0; i < 3; i++ {
			if len(a[i]) != c {
				return fmt.Errorf("TersPrintMatrix: inconsistant row lenth")
			}
			TersPrintArray(a[i])
		}
		for i := 0; i < 2; i++ {
			fmt.Printf("		.				.				.\n")
		}
		for i := 3; i > 0; i-- {
			if len(a[r-i]) != c {
				return fmt.Errorf("TersPrintMatrix: inconsistant row lenth")
			}
			TersPrintArray(a[r-i])
		}
	} else {
		for i := 0; i < r; i++ {
			if len(a[i]) != c {
				return fmt.Errorf("TersPrintMatrix: inconsistant row lenth")
			}
			TersPrintArray(a[i])
		}
	}
	fmt.Printf("]\n")
	return nil
}

/*
LPSimplexVerboseCallback is an example callbase with verbose output.
A sample callback function demonstrating the LPSimplex callback interface.
This callback produces detailed output to sys.stdout before each
iteration and after the final iteration of the simplex algorithm.

    Parameters
    ----------
    xk : array_like (float64)
       The current solution vector.
    tableau : array_like
        The current tableau of the simplex algorithm.
        Its structure is defined in _solve_simplex.
    phase : int
        The current Phase of the simplex algorithm (1 or 2)
    nit : int
        The current iteration number.
    pivcol : (int)
		The column index of the tableau selected as the next
		pivot column or -1 if no pivot exists
    pivrow : (int)
		The row index of the tableau selected as the next pivot
		row or -1 if no pivot exists
    basis : array(int)
        A list of the current basic variables.
        Each element contains the name of a basic variable and its value.
    complete : bool
        True if the simplex algorithm has completed
        (and this is the final call to callback), otherwise False.
*/
func LPSimplexVerboseCallback(xk []float64, tableau [][]float64, nit, pivrow, pivcol, phase int, basis []int, complete bool) {

	if complete {
		fmt.Printf("--------- Iteration Complete - Phase %d -------\n", phase)
		fmt.Printf("Tableau:\n")
	} else if nit == 0 {
		fmt.Printf("--------- Initial Tableau - Phase %d ----------\n", phase)
	} else {
		fmt.Printf("--------- Iteration %d  - Phase %d --------\n", nit, phase)
		fmt.Printf("Tableau:\n")
	}

	if nit >= 0 {
		err := TersPrintMatrix(tableau)
		if err != nil {
			fmt.Printf("Error - %s\n", err)
		}
		if !complete {
			fmt.Printf("Pivot Element: T[%d, %d]\n", pivrow, pivcol)
		}
		//fmt.Printf("Basic Variables: %v\n", basis)
		fmt.Printf("Basic Variables:\n")
		TersPrintIntArray(basis)
		fmt.Printf("\n")
		fmt.Printf("Current Solution:\n")
		fmt.Printf("x = \n")
		TersPrintArray(xk)
		fmt.Printf("\n")
		fmt.Printf("Current Objective Value:\n")
		fmt.Printf("f = %f\n", -tableau[len(tableau)-1][len(tableau[0])-1])
		fmt.Printf("\n")
	}
}

/*
LPSimplexTerseCallback is an example callbase with verbose output.
	A sample callback function demonstrating the LPSimplex callback interface.
	This callback produces brief output to sys.stdout before each iteration
	and after the final iteration of the simplex algorithm.

	Parameters
	----------
	xk : array_like (float64)
	    The current solution vector.
	tableau : matrix_like (float64 x float64)
		The current tableau of the simplex algorithm.
		Its structure is defined in _solve_simplex.
	nit : int
		The current iteration number.
	pivcol : (int)
		The column index of the tableau selected as the next
		pivot column or -1 if no pivot exists
	pivrow : (int)
		The row index of the tableau selected as the next pivot
		row or -1 if no pivot exists
	phase : int
		The current Phase of the simplex algorithm (1 or 2)
	basis : list[tuple(int, float)]
		A list of the current basic variables.
		Each element contains the index of a basic variable and
		its value.
	complete : bool
		True if the simplex algorithm has completed
		(and this is the final call to callback), otherwise False.
*/
func LPSimplexTerseCallback(xk []float64, tableau [][]float64, nit, pivrow, pivcol, phase int, basis []int, complete bool) {

	if nit == 0 {
		fmt.Printf("Iter:   X:\n")
	}
	fmt.Printf("%d   x = %v\n", nit, xk)
}

/*
getPivotCol searches a simplex tableu for best variable to enter.
	Given a linear programming simplex tableau, determine the column
	of the variable to enter the basis.

	Parameters
	----------
	T : 2D array (float64xfloat64)
	    The simplex tableau.
	tol : float
	    Elements in the objective row larger than -tol will not be considered
	    for pivoting.  Nominally this value is zero, but numerical issues
	    cause a tolerance about zero to be necessary.
	bland : bool
	    If True, use Bland's rule for selection of the column (select the
	    first column with a negative coefficient in the objective row,
	    regardless of magnitude).

	Returns
	-------
	col: int
	    The index of the column of the pivot element.
	    If no column is found, col will be returned as -1.
*/
func getPivotCol(T [][]float64, tol float64, bland bool) int {

	var min float64
	col := -1
	for j := 0; j < len(T[0])-1; j++ {
		if T[len(T)-1][j] < -tol {
			if col < 0 || T[len(T)-1][j] < min {
				min = T[len(T)-1][j]
				col = j
				if bland {
					return j // this is the first column
				}
			}
		}
	}
	return col
}

/*
getPivotRow searches simplex tableau for best row to pivot.
	Given a linear programming simplex tableau, determine the row for the
	pivot operation.

	Parameters
	----------
	T : 2D array
		The simplex tableau.
	pivcol : int
		The index of the pivot column.
	phase : int
		The phase of the simplex algorithm (1 or 2).
	tol : float
		Elements in the pivot column smaller than tol will not be considered
		for pivoting.  Nominally this value is zero, but numerical issues
		cause a tolerance about zero to be necessary.

	Returns
	-------
	row: int
		The index of the row of the pivot element.  If no row is found, row
		will be returned as -1.
*/
func getPivotRow(T [][]float64, pivcol int, phase int, tol float64) int {
	var qmin float64
	var k int
	row := -1
	if phase == 1 {
		k = 2
	} else {
		k = 1
	}
	for i := 0; i < len(T)-k; i++ {
		if T[i][pivcol] > tol { // WGA if b[i]>=0 then this check is enough
			q := T[i][len(T[0])-1] / T[i][pivcol]
			if row < 0 || q < qmin {
				// Bland's rule requires the first i is used in case of
				// a tie for qmin, the above if satisfies the requirement
				qmin = q
				row = i
			}
		}
	}
	return row
}

func doPivot(T [][]float64, basis []int, pivrow, pivcol int) {
	// variable represented by pivcol enters
	// variable in basis[pivrow] leaves
	basis[pivrow] = pivcol
	pivval := T[pivrow][pivcol]
	//if pivval != 1.0 { // experimental: row will not change so do nothing (slows things down :-( ))
	for j := 0; j < len(T[0]); j++ {
		T[pivrow][j] = T[pivrow][j] / pivval
	}
	//}
	for irow := range T {
		if irow != pivrow {
			mul := T[irow][pivcol]
			if mul != 0.0 { // experimental: row will not change so do nothing (large speed up ;-) )
				for j := 0; j < len(T[0]); j++ {
					T[irow][j] = T[irow][j] - (T[pivrow][j] * mul)
				}
			}
		}
	}
}

/*
solveSimplex does the actual work to find the optimal LP solution.
	Solve a linear programming problem in "standard maximization form" using
	the Simplex Method.

	Minimize :math:`f = c^T x`

	subject to
		    	Ax = b
		        x_i >= 0
		        b_j >= 0

	Parameters
	----------
	T : matrix_like (float64 x float64)
		A 2-D array representing the simplex T corresponding to the
		maximization problem.  It should have the form:

		       [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
		        [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
		        .
		        .
		        .
		        [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
		        [c[0],   c[1], ...,   c[n_total],    0]]

		        However, the phase 1 starting point uses:

		       [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
		        [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
		        .
		        .
		        .
		        [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
		        [c[0],   c[1], ...,   c[n_total],   0],
		        [c'[0],  c'[1], ...,  c'[n_total],  0]]

				(a Problem in which a basic feasible
				solution is sought prior to maximizing the actual objective.
				T is modified in place by solveSimplex.
	n : int
		The number of true variables in the problem.
	basis : array
		An array of the indices of the basic variables, such that
		basis[i] contains the column corresponding to the basic
		variable for row i.
		Basis is modified in place by solveSimplex
	maxiter : int
		The maximum number of iterations to perform before aborting the
		optimization.
	phase : int
		The phase of the optimization being executed.  In phase 1
		a basic feasible solution is sought and the T has an
		additional row representing an alternate objective function.
	callback : callable
		If a callback function is provided, it will be called within
		each iteration of the simplex algorithm. The callback must
		Callbackfunc is the taking arguments:
			"xk" : the current solution vector
			"T" : The current Simplex algorithm T
			"nit" : The current iteration.
			"pivot" : The pivot (row, column) used for the next iteration.
			"phase" : Whether the algorithm is in Phase 1 or Phase 2.
			"basis" : The indices of the columns of the basic variables.
	tol : float
		The tolerance which determines when a solution is "close
		enough" to zero in Phase 1 to be considered a basic feasible
		solution or close enough to positive to to serve as an optimal
		solution.
	nit0 : int
		The initial iteration number used to keep an accurate iteration
		total in a two-phase problem.
	bland : bool
		If True, choose pivots using Bland's rule [3].  In problems
		which fail to converge due to cycling, using Bland's rule can
		provide convergence at the expense of a less optimal path about
		the simplex.

	Returns
	-------
	res : OptResult
		Important attributes are: ``x`` the solution array, ``success``
		a Boolean flag indicating if the optimizer exited successfully
		and ``message`` which describes the cause of the termination.
		Possible values for the ``status`` attribute are:
		    0 : Optimization terminated successfully
		    1 : Iteration limit reached
		    2 : Problem appears to be infeasible
		    3 : Problem appears to be unbounded
*/
func solveSimplex(T [][]float64, n int, basis []int, maxiter int, phase int,
	callback Callbackfunc, tol float64, nit0 int, bland bool) (
	nit int, status int) {

	lastObjectiveValue = 0.0
	// tol=1.0E-12
	nit = nit0
	complete := false
	userBland := bland // track the orig bland setting
	var m int

	if phase == 1 {
		//m = T.shape[0]-2
		m = len(T) - 2
	} else if phase == 2 {
		//m = T.shape[0]-1
		m = len(T) - 1
	} else {
		fmt.Printf("Argument 'phase' to _solve_simplex must be 1 or 2\n")
		os.Exit(1)
	}

	if phase == 2 {
		// Check if any artificial variables are still in the basis.
		// If yes, check if any coefficients from this row and a column
		// corresponding to one of the non-artificial variable is non-zero.
		// If found, pivot at this term. If not, start phase 2.
		// Do this for all artificial variables in the basis [4].
		// Ref: "An Introduction to Linear Programming and Game Theory"
		// by Paul R. Thie, Gerard E. Keough, 3rd Ed,
		// Chapter 3.7 Redundant Systems (pag 102)
		rows := make([]int, 0)
		for row := 0; row < len(basis); row++ { //mimic the following comprehension
			if basis[row] > len(T[0])-2 {
				rows = append(rows, row)
			}
		}
		for r := range rows {
			pivrow := rows[r]
			non_zero_row := make([]int, 0)
			for col := 0; col < len(T[0])-1; col++ {
				if T[pivrow][col] != 0 {
					non_zero_row = append(non_zero_row, col)
				}
			}
			if len(non_zero_row) > 0 {
				pivcol := non_zero_row[0]
				// variable represented by pivcol enters
				// variable in basis[pivrow] leaves
				basis[pivrow] = pivcol
				pivval := T[pivrow][pivcol]
				for j := 0; j < len(T[0]); j++ {
					T[pivrow][j] = T[pivrow][j] / pivval
				}
				for irow := 0; irow < len(T); irow++ {
					if irow != pivrow {
						mul := T[irow][pivcol]
						for j := 0; j < len(T[0]); j++ {
							T[irow][j] = T[irow][j] - (T[pivrow][j] * mul)
						}
					}
				}
				nit++
			}
		}
	}

	var solution []float64
	ll := len(basis[:m])
	if ll == 0 { // WGA TODO why is this not alway m??
		//fmt.Printf("HELP (line 410): ll: %d\n", ll)
		solution = make([]float64, len(T[0])-1) //np.zeros(T.shape[1] - 1, dtype=np.float64)
	} else {
		//WGA TODO double check this for correctness
		size := max(len(T[0])-1, maxlist(basis[:m])+1)
		//fmt.Printf("HELP (line 414): ll: %d size: %d\n", ll, size)
		solution = make([]float64, size)
		//np.zeros(max(T.shape[1] - 1, max(basis[:m]) + 1),
		//                   dtype=np.float64)
	}

	for {
		if complete {
			break
		}

		var pivrow int
		pivcol := getPivotCol(T, tol, bland) // bland past in or adjusted below
		if pivcol < 0 {
			pivcol = -1 // invalue value math.NaN()
			pivrow = -1 // invalue value math.NaN()
			status = 0
			complete = true
		} else {
			pivrow = getPivotRow(T, pivcol, phase, tol)
			if pivrow < 0 {
				//fmt.Printf("***** Pivot col: %d ******\n", pivcol)
				status = 3
				complete = true
			}
		}
		if callback != nil {
			for i := 0; i < len(solution); i++ {
				solution[i] = 0
			}
			for i := 0; i < m; i++ {
				solution[basis[i]] = T[i][len(T[0])-1]
			}
			callback(solution[:n], T, nit, pivrow, pivcol,
				phase, basis, (complete && phase == 2))
		}

		if !complete {
			if nit >= maxiter {
				status = 1
				complete = true
			} else {
				doPivot(T, basis, pivrow, pivcol)
				nit++
				if true { // experiment
					bland = userBland // reset to orig bland setting
					objValue := -T[len(T)-1][len(T[0])-1]
					if objValue == lastObjectiveValue {
						degeneritePivotCount++
						if dynamicBlandRule {
							//fmt.Printf("Starting Dynamic Bland Rule: dpc: %d, pivot count: %d\n", degeneritePivotCount, nit)
							bland = true
						}
					}
					lastObjectiveValue = objValue
				}
			}
		}
	}
	return nit, status
}

// Bound is a struc that carries the low and high values of a variable's range.
type Bound struct {
	lb float64
	ub float64
}

/*
LPSimplex takes a general LP model, converts to standard LP and then
calls solveSimplex() to solves each phase of the two-phase algorithm.

	Solve the following linear programming problem via a two-phase
	simplex algorithm.

	minimize:     c^T * x

	subject to:   A_ub * x <= b_ub
		          A_eq * x == b_eq
		          x >= 0 if not explicitly overriden with defined bounds

	Parameters
	----------
	c : array_like
		Coefficients of the linear objective function to be minimized.
	A_ub : array_like
		2-D array which, when matrix-multiplied by x, gives the values of the
		upper-bound inequality constraints at x.
	b_ub : array_like
		1-D array of values representing the upper-bound of each inequality
		constraint (row) in A_ub.
	A_eq : array_like
		2-D array which, when matrix-multiplied by x, gives the values of the
		equality constraints at x.
	b_eq : array_like
		1-D array of values representing the RHS of each equality constraint
		(row) in A_eq.
	bounds : array_like of Bound struct with lb, ub of type float64
		The bounds for each independent variable in the solution, which
		can take one of three forms::
		    None : The default bounds, all variables are non-negative.
			(lb, ub) : if only one struct, the same lower bound (lb) and
			   		   upper bound (ub) will be applied to all variables.
			[(lb_0, ub_0), (lb_1, ub_1), ...] : If an n x Bound struct
					   sequence is provided, each variable x_i will be
					   bounded by Bounds[i].lb and Bound[i].ub.
		    Infinite bounds are specified using math.Inf(-1) (negative)
		       		   and math.Inf(1) (positive).
	callback : callable
		If a callback function is provide, it will be called within each
		iteration of the simplex algorithm. The callback must have the
		signature `callback(xk, **kwargs)` where xk is the current solution
		vector and kwargs is a dictionary containing the following::
		    "tableau" : The current Simplex algorithm tableau
		    "nit"     : The current iteration.
		    "pivot"   : The pivot (row, column) used for the next iteration.
		    "phase"   : Whether the algorithm is in Phase 1 or Phase 2.
			"bv"      : A structured array containing a string representation
			            of each basic variable and its current value.
	maxiter : int
		The maximum number of iterations to perform.
	disp : bool
		If True, print exit status message to sys.stdout
	tol : float
		The tolerance which determines when a solution is "close enough" to zero
		in Phase 1 to be considered a basic feasible solution or close enough
		to positive to to serve as an optimal solution.
	bland : bool
		If True, use Bland's anti-cycling rule [3] to choose pivots to
		prevent cycling.  If False, choose pivots which should lead to a
		converged solution more quickly.  The latter method is subject to
		cycling (non-convergence) in rare instances.

	Returns
	-------
	An OptResult struct consisting of the following fields::
		x : []float64
		    The independent variable vector which optimizes the linear
		    programming problem.
		fun : float64
		    Value of the objective function.
		slack : []float64
		    The values of the slack variables.  Each slack variable corresponds
		    to an inequality constraint.  If the slack is zero, then the
		    corresponding constraint is active.
		success : bool
		    Returns True if the algorithm succeeded in finding an optimal
		    solution.
		status : int
		    An integer representing the exit status of the optimization::
		    0 : Optimization terminated successfully
		    1 : Iteration limit reached
		    2 : Problem appears to be infeasible
		    3 : Problem appears to be unbounded
		nit : int
		    The number of iterations performed.
		message : str
		    A string descriptor of the exit status of the optimization.

	Examples
	--------
	Consider the following problem:

	Minimize: f = -1*x[0] + 4*x[1]

	Subject to: -3*x[0] + 1*x[1] <= 6
		         1*x[0] + 2*x[1] <= 4
		                    x[1] >= -3

		   where:  -inf <= x[0] <= inf

	This problem deviates from the standard linear programming problem.  In
	standard form, linear programming problems assume the variables x are
	non-negative.  Since the variables don't have standard bounds where
	0 <= x <= inf, the bounds of the variables must be explicitly set.

	There are two upper-bound constraints, which can be expressed as

	dot(A_ub, x) <= b_ub

	The input for this problem is as follows:

	>>> from scipy.optimize import linprog
	>>> c = [-1, 4]
	>>> A = [[-3, 1], [1, 2]]
	>>> b = [6, 4]
	>>> x0_bnds = (None, None)
	>>> x1_bnds = (-3, None)
	>>> res = linprog(c, A, b, bounds=(x0_bnds, x1_bnds))
	>>> print(res)
		        fun: -22.0
	message: 'Optimization terminated successfully.'
		        nit: 1
		      slack: array([ 39.,   0.])
		     status: 0
		    success: True
		          x: array([ 10.,  -3.])

	References
	----------
	[1] Dantzig, George B., Linear programming and extensions. Rand
		Corporation Research Study Princeton Univ. Press, Princeton, NJ, 1963
	[2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to
		Mathematical Programming", McGraw-Hill, Chapter 4.
	[3] Bland, Robert G. New finite pivoting rules for the simplex method.
		Mathematics of Operations Research (2), 1977: pp. 103-107.
	[4] Paul R. Thie, Gerard E. Keough, "An Introduction to Linear
		Programming and Game Theory" 3rd Ed, Chapter 3.7 Redundant Systems
		pp. 102
*/
func LPSimplex(cc []float64,
	Aub [][]float64, bub []float64,
	Aeq [][]float64, beq []float64,
	bounds []Bound, callback Callbackfunc, disp bool,
	maxiter int, tol float64, bland bool) OptResult {

	//tol=1.0E-12
	var message string
	status := 0
	messages := [...]string{
		"Optimization terminated successfully.",
		"Iteration limit reached.",
		"Optimization failed. Unable to find a feasible starting point.",
		"Optimization failed. The problem appears to be unbounded.",
		"Optimization failed. Singular matrix encountered."}

	have_floor_variable := false

	// The initial value of the objective function element in the tableau
	f0 := float64(0)

	// The number of variables as given by c
	n := len(cc)

	// Analyze the bounds and determine what modifications to be made to
	// the constraints in order to accommodate them.
	L := make([]float64, n)
	U := make([]float64, n)
	for i := 0; i < n; i++ {
		U[i] = math.Inf(0)
	}

	if bounds == nil || len(bounds) == 0 {
		// do nothing
	} else if len(bounds) == 1 {
		// All bounds are the same
		for i := 0; i < n; i++ {
			L[i] = bounds[0].lb
			U[i] = bounds[0].ub
		}
	} else {
		if len(bounds) != n {
			status = -1
			message = "Invalid input for LPSimplex. Length of bounds is inconsistent with the length of c"
		} else {
			for i := 0; i < n; i++ {
				L[i] = bounds[i].lb
				U[i] = bounds[i].ub
			}
		}
	}

	if anyNegInf(L) {
		// If any lower-bound constraint is a free variable
		// add the first column variable as the "floor" variable which
		// accommodates the most negative variable in the problem.
		n = n + 1
		L = append([]float64{0}, L...)
		U = append([]float64{0}, U...)
		cc = append([]float64{0}, cc...)
		for i := 0; i < len(Aeq); i++ {
			Aeq[i] = append([]float64{0}, Aeq[i]...)
		}
		for i := 0; i < len(Aub); i++ {
			Aub[i] = append([]float64{0}, Aub[i]...)
		}
		have_floor_variable = true
	}

	/*
	   Now before we deal with any variables with lower bounds < 0,
	   deal with finite positive bounds which can be simply added as
	   new constraints. Also validate bounds inputs here.
	*/
	for i := 0; i < n; i++ {
		if L[i] > U[i] {
			status = -1
			message = fmt.Sprintf("Invalid input for LPSimplex.  Lower bound var %d is greater than upper bound var %d", i, i)
		}

		if math.IsInf(L[i], 1) {
			status = -1
			message = "Invalid input for LPSimplex " +
				"Lower bound may not be +infinity"
		} else if L[i] > 0 {
			// Add a new lower-bound (negative upper-bound) constraint
			newConstraint := make([]float64, n)
			Aub = append(Aub, newConstraint)
			Aub[len(Aub)-1][i] = -1
			bub = append(bub, -L[i])
			L[i] = 0
		}

		if math.IsInf(U[i], -1) {
			status = -1
			message = "Invalid input for LPSimplex'.  Upper bound may not be -infinity"
		} else if !math.IsInf(U[i], 1) {
			// Add a new upper-bound constraint
			newConstraint := make([]float64, n)
			Aub = append(Aub, newConstraint)
			Aub[len(Aub)-1][i] = 1
			bub = append(bub, U[i])
			U[i] = math.Inf(1)
		}
	}

	/*
	   Now find negative lower bounds (finite or infinite) which require a
	   change of variables or free variables and handle them appropriately
	*/
	for i := 0; i < n; i++ {
		if L[i] < 0 {
			if !math.IsInf(L[i], 0) && L[i] < 0 {
				// Add a change of variables for x[i]
				// For each row in the constraint matrices, we take the
				// coefficient from column i in A,
				// and subtract the product of that and L[i] to the RHS b
				for r := 0; r < len(Aeq); r++ {
					beq[r] = beq[r] - (Aeq[r][i] * L[i])
				}
				for r := 0; r < len(Aub); r++ {
					bub[r] = bub[r] - (Aub[r][i] * L[i])
				}
				// We now have a nonzero initial value for the objective
				// function as well.
				f0 = f0 - cc[i]*L[i]
			} else {
				// This is an unrestricted variable, let x[i] = u[i] - v[0]
				// where v is the first column in all matrices.
				for r := 0; r < len(Aeq); r++ {
					Aeq[r][0] = Aeq[r][0] - Aeq[r][i]
				}
				for r := 0; r < len(Aeq); r++ {
					Aub[r][0] = Aub[r][0] - Aub[r][i]
				}
				cc[0] = cc[0] - cc[i]
			}
		}
		if math.IsInf(U[i], -1) {
			status = -1
			message = "Invalid input for LPSimplex. Upper bound may not be -inf."
		}
	}
	/*
	 */

	// The number of upper bound constraints (rows in A_ub and elements in b_ub)
	mub := len(bub)

	// The number of equality constraints (rows in A_eq and elements in b_eq)
	meq := len(beq)

	// The total number of constraints
	m := mub + meq

	// The number of slack variables (one for each inequality)
	n_slack := mub

	// The number of artificial variables (one for each lower-bound and equality
	// constraint)
	// FIXME: Here we include an artificial var for each RHS b value that is
	// negative. Robert J. Vanderbei in Linear Programming: Foundations and
	// Extentions, Fourth Edition shows how this can be done with a single
	// added variable.
	n_artificial := meq + countNegEntries(bub)

	Aub_rows, Aub_cols, err := checkRectangle(Aub)
	if err != nil {
		fmt.Printf("A_ub Matrix error: %v\n", err)
	}

	Aeq_rows, Aeq_cols, err := checkRectangle(Aeq)
	if err != nil {
		fmt.Printf("A_eq Matrix error: %v\n", err)
	}

	if Aeq_rows != meq {
		status = -1
		message = "Invalid input for LPSimplex.  " +
			"The number of rows in A_eq must be equal " +
			"to the number of values in b_eq"
		fmt.Printf("%s\n", message)
	}

	if Aub_rows != mub {
		status = -1
		message = "Invalid input for LPSimplex. " +
			"The number of rows in A_ub must be equal " +
			"to the number of values in b_ub"
		fmt.Printf("%s\n", message)
	}

	if Aeq_cols > 0 && Aeq_cols != n {
		status = -1
		message = "Invalid input for LPSimplex.  " +
			"Number of columns in A_eq must be equal " +
			"to the size of c"
		fmt.Printf("%s\n", message)
	}

	if Aub_cols > 0 && Aub_cols != n {
		status = -1
		message = "Invalid input for LPSimplex.  " +
			"Number of columns in A_ub must be equal to the size of c"
		fmt.Printf("%s\n", message)
	}

	if status != 0 {
		// Invalid inputs provided
		os.Exit(1)
	}

	// FIXME: if I want to offer anonther entry point that does not
	// include setting bounds, this may be a good place to start the
	// new function so that the constaints can be passed in, same as
	// passed to LPSimplex but not bounds. Needs some thought.
	//
	// This might also be a place where a revised simplex inplementation
	// could be added / substitued.
	//
	// This may also be a good spot to include call to the scaling
	// routine. Give it a try here:

	// Create the tableau
	T := make([][]float64, m+2)
	for i := 0; i < m+2; i++ {
		T[i] = make([]float64, n+n_slack+n_artificial+1)
	}
	// Insert objective into tableau
	for j, val := range cc {
		T[len(T)-2][j] = val //Aeq[i][j]
	}
	T[len(T)-2][len(T[0])-1] = f0

	if meq > 0 {
		//# Add Aeq to the tableau
		for i, row := range Aeq {
			for j, val := range row {
				T[i][j] = val //Aeq[i][j]
			}
		}
		// Add beq to the tableau
		for i, val := range beq {
			T[i][len(T[0])-1] = val //beq[i]
		}
	}
	if mub > 0 {
		// Add Aub to the tableau
		for i, row := range Aub {
			for j, val := range row {
				T[meq+i][j] = val //Aub[i][j]
			}
		}
		// At bub to the tableau
		for i, val := range bub {
			T[meq+i][len(T[0])-1] = val //bub[i]
		}
		// Add the slack variables to the tableau
		for i := 0; i < m-meq; i++ {
			for j := 0; j < n_slack; j++ {
				if i == j {
					T[meq+i][n+j] = float64(1)
				}
			}
		}
	}

	// Further set up the tableau.
	// If a row corresponds to an equality constraint or a negative b
	// (a lower bound constraint), then an artificial variable is added
	// for that row. Also, if b is negative, first flip the signs in
	// that constraint.
	slcount := 0
	avcount := 0
	basis := make([]int, m)
	r_artificial := make([]int, n_artificial)
	for i := 0; i < m; i++ {
		if i < meq || T[i][len(T[0])-1] < 0 {
			// basic variable i is in column n+n_slack+avcount
			basis[i] = n + n_slack + avcount
			r_artificial[avcount] = i
			avcount += 1
			if T[i][len(T[0])-1] < 0 { // b[i] is negative
				for j := range T[i] {
					T[i][j] *= float64(-1)
				}
			}
			T[i][basis[i]] = float64(1)
			T[len(T)-1][basis[i]] = float64(1)
		} else {
			// basic variable i is in column n+slcount
			basis[i] = n + slcount
			slcount += 1
		}
	}
	// Make the artificial variables basic feasible variables by subtracting
	// each row with an artificial variable from the Phase 1 objective
	for i := range r_artificial {
		r := r_artificial[i]
		for j := 0; j < len(T[0]); j++ {
			T[len(T)-1][j] = T[len(T)-1][j] - T[r][j]
		}
	}

	degeneritePivotCount = 0
	phase := 1
	nit0 := 0
	nit1, status := solveSimplex(T, n, basis, maxiter, phase, callback, tol, nit0, bland)

	// if pseudo objective is zero, remove the last row from the tableau and
	// proceed to phase 2
	if math.Abs(T[len(T)-1][len(T[0])-1]) <= tol {
		// Remove the pseudo-objective row from the tableau
		T = T[:len(T)-1]
		// Remove the artificial variable columns from the tableau
		for i := 0; i < len(T); i++ {
			T[i] = append(T[i][:n+n_slack], T[i][n+n_slack+n_artificial:]...)
		}
	} else {
		// Failure to find a feasible starting point
		status = 2
	}

	if status != 0 {
		message = messages[status]
		if disp {
			fmt.Printf("%s\n", message)
		}
		obj := -T[len(T)-1][len(T[0])-1]
		return OptResult{[]float64{math.NaN()},
			obj, nit1,
			status, nil, message, false}
	}

	// Phase 2
	phase = 2
	maxiter -= nit1
	nit2, status := solveSimplex(T, n, basis, maxiter, phase, callback,
		tol, nit1, bland)

	solution := make([]float64, n+n_slack+n_artificial)
	for i := 0; i < m; i++ {
		solution[basis[i]] = T[i][len(T[0])-1]
	}
	x := solution[:n]
	slack := solution[n : n+n_slack]

	for i := 0; i < n; i++ {
		if !math.IsInf(L[i], -1) {
			x[i] = x[i] + L[i]
		}
	}

	// For those variables with infinite negative lower bounds,
	// take x[i] as the difference between x[i] and the floor variable.
	if have_floor_variable {
		for i := 1; i < n; i++ {
			if math.IsInf(L[i], -1) {
				x[i] -= x[0]
			}
		}
		x = x[1:]
	}

	// Optimization complete at this point
	obj := -T[len(T)-1][len(T[0])-1]
	if status == 1 || status == 0 {
		if disp {
			fmt.Printf("%s\n", messages[status])
			fmt.Printf("         Current function value: %12.6f\n", obj)
			fmt.Printf("         Iterations: %d\n", nit2)
		}
	} else {
		if disp {
			fmt.Printf("%s\n", messages[status])
			fmt.Printf("         Iterations: %d\n", nit2)
		}
	}

	return OptResult{x, obj, nit2,
		status, slack,
		messages[status],
		(status == 0)}
}

// countNegEntries counts the number of array entries that are less than zero.
func countNegEntries(A []float64) int {
	if A == nil || len(A) < 1 {
		return 0
	}
	n := 0
	for _, e := range A {
		if e < 0 {
			n++
		}
	}
	return n
}

// anyNegInf searches array for negitive infinity entries.
func anyNegInf(A []float64) bool { // WGA TEST ME TODO
	for _, e := range A {
		if math.IsInf(e, -1) {
			return true
		}
	}
	return false
}

// checkRectangle checks for uniformity and returns the number of rows and columns.
func checkRectangle(A [][]float64) (rows int, cols int, err error) {

	if A == nil {
		rows = 0
		cols = 0
		err = nil
		return
	}
	rows = len(A)
	if rows > 0 {
		cols = len(A[0])
		if cols <= 1 {
			return 0, 0, fmt.Errorf("Invalid input, must be two-dimensional")
		}
		for i := 1; i < rows; i++ {
			if cols != len(A[i]) {
				return 0, 0, fmt.Errorf("Invalid input, all rows must have the same length")
			}
		}
	} else {
		return 0, 0, fmt.Errorf("Invalid input, must be two-dimensional")
	}
	return rows, cols, nil
}

// max returns the larger of two integers.
func max(a, b int) int { //TODO add error checking
	lmax := b
	if a > b {
		lmax = a
	}
	return lmax
}

// maxlist returns the largest integer in a list of integers.
func maxlist(list []int) int { //TODO add error checking
	lmax := list[0]
	for e := range list[1:] {
		if e > lmax {
			lmax = e
		}
	}
	return lmax
}
