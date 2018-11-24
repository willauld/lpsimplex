package lpsimplex

import (
	"fmt"
	"math"
	"testing"
)

func TestScaling(t *testing.T) {
	A, b, c := GetModelSmall_1()
	tests := []struct {
		// min cx s.t. (a:ae)x <= b:be
		a      [][]float64
		b      []float64
		c      []float64
		ae     [][]float64
		be     []float64
		bounds []Bound
		x      []float64
		opt    float64
		intr   int
		errstr string
	}{
		// Basic feasible LP
		// Case 0
		{[][]float64{{-1, 2, 1, 0}, {3, 1, 0, 1}},
			[]float64{4, 9},
			[]float64{-1, -2, 0, 0},
			nil,
			nil,
			[]Bound{},
			[]float64{2, 2, 0, 0},
			-8,
			2,
			"",
		},
		// Case 1
		{[][]float64{{-3, 1}, {1, 2}},
			[]float64{6, 4},
			[]float64{-1, 4},
			nil,
			nil,
			[]Bound{},
			[]float64{0, 0},
			-4,
			1,
			"",
		},
		// Case 2
		{[][]float64{{-3, 1}, {1, 2}},
			[]float64{6, 4},
			[]float64{-1, 4},
			nil,
			nil,
			[]Bound{{math.Inf(-1), math.Inf(1)}, {-3, math.Inf(1)}},
			[]float64{0, 0},
			-22,
			1,
			"",
		},
		// Case 3
		// Taken from an Introduction to Linear Programming and Game Theory, Thie and Keough
		{[][]float64{{4, -1, 0, 1}, {7, -8, -1, 0}}, //Ch. 3, page 59
			[]float64{6, -7},
			[]float64{-3, 2, 1, -1},
			[][]float64{{1, 1, 0, 4}},
			[]float64{12},
			[]Bound{
				{0, math.Inf(1)},
				{0, math.Inf(1)},
				{0, math.Inf(1)},
				{math.Inf(-1), math.Inf(1)},
			},
			[]float64{0, 0, 0, 0},
			-2.235294117647059,
			3,
			"",
		},
		// Case 4
		{nil, // hand converted case 3
			nil,
			[]float64{-3, 2, 1, -1, 1, 0, 0},
			[][]float64{{4, -1, 0, 1, -1, 1, 0}, {-7, 8, 1, 0, 0, 0, -1}, {1, 1, 0, 4, -4, 0, 0}}, //Ch. 3, page 59
			[]float64{6, 7, 12},
			[]Bound{},
			[]float64{0, 0, 0, 0},
			-2.235294117647059,
			5, // Iterations with scaling
			// 3, // Iteractions without scalling
			"",
		},
		// Case 5
		{A,
			b,
			c,
			nil,
			nil,
			[]Bound{},
			[]float64{2, 2, 0, 0},
			-490152.548580589006, // This is what the scaled version gets
			// -490152.548580589821, // This is what the non-scaled version gets
			79, // Iterations with scaling
			//77, // Iterations without scaling
			"",
		},
		// Case 6
		{[][]float64{{3, 2, 0}, {-1, 1, 4}, {2, -2, 5}},
			[]float64{60, 10, 50},
			[]float64{-2, -3, -3},
			nil,
			nil,
			[]Bound{},
			[]float64{8, 18, 70},
			-70,
			2,
			"",
		},
		// Case 7
		{[][]float64{{1, 1, -2}, {-3, 1, 2}},
			[]float64{7, 3},
			[]float64{0, -2, -1},
			nil,
			nil,
			[]Bound{},
			[]float64{1, 6, 0},
			-12,
			2,
			"Optimization failed. The problem appears to be unbounded.",
		},
		// Case 8
		{nil,
			nil,
			[]float64{-4, 1, 30, -11, -2, 3, 0},
			[][]float64{{-2, 0, 6, 2, 0, -3, 1}, {-4, 1, 7, 1, 0, -1, 0}, {0, 0, -5, 3, 1, -1, 0}},
			[]float64{20, 10, 60},
			[]Bound{},
			[]float64{1, 6, 0},
			-230.0,
			5, // Iterations with scaling
			//4, // Iterations without scaling
			"",
		},
		// Case 9
		{[][]float64{{-1, 2, 1}, {1, 0, -2}},
			[]float64{1, -4},
			[]float64{1, 1, 1},
			[][]float64{{1, -1, 2}},
			[]float64{4},
			[]Bound{},
			[]float64{2 / 3, 0, 5 / 3},
			1.333333333333,
			2,
			"Optimization failed. Unable to find a feasible starting point.",
		},
	}

	tol := 1.0E-12
	//tol := float64(0)
	bland := false
	maxiter := 4000
	//bounds := []Bound(nil)
	//callback := LPSimplexVerboseCallback
	//callback := LPSimplexTerseCallback
	callback := Callbackfunc(nil)
	disp := false //true

	for i, elem := range tests {
		fmt.Printf("============== Case %d ==============\n", i)
		LPSimplexSetNewBehavior(NB_CMD_RESET | NB_CMD_SCALEME)
		res := LPSimplex(elem.c, elem.a, elem.b, elem.ae, elem.be, elem.bounds, callback, disp, maxiter, tol, bland)
		//fmt.Printf("Res: %+v\n", res)
		//fmt.Printf("Case %d returned with success value of %v and objective value %v\n", i, res.Success, res.Fun)
		if !res.Success {
			if elem.errstr != res.Message {
				t.Errorf("TestLinprog Case %d: failed with message %s\n", i, res.Message)
			}
		}
		if math.Abs(elem.opt-res.Fun) > tol {
			t.Errorf("TestLinprog Case %d: Fun: %.12f but expected %.12f\n", i, res.Fun, elem.opt)
		}
		if elem.intr != res.Nitr {
			t.Errorf("TestLinprog Case %d: Nitr: %d but expected %d\n", i, res.Nitr, elem.intr)
		}
		e, str := doubleVectorEquals(elem.x, res.X)
		if e {
			t.Errorf("TestLinprog Case %d: x: %v but expected: %v\n\tReason: %s", i, res.X, elem.x, str)
		}
	}
}
