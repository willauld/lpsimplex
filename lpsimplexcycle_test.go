package lpsimplex

import (
	"fmt"
	"testing"
	"time"
	"math"
)

func TestCycleCases(t *testing.T) {
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
		useBland bool
		dynamicBland bool 
	}{
		{
			// Basic feasible LP
			// Case 0
			[][]float64{{-1, 2, 1, 0}, {3, 1, 0, 1}},
			[]float64{4, 9},
			[]float64{-1, -2, 0, 0},
			nil,
			nil,
			[]Bound{},
			[]float64{2, 2, 0, 0},
			-8,
			2,
			"",
			false,
			false,
		},
		{
			// Basic feasible Cycling LP
			// Example from Robert J. Vanderbei, 
			// Linear Programming: Foundations and Extensions Fourth Edition
			// page 26 (bottom of page)
			// Example is maximization so need to use -c[] 
			// Case 1
			[][]float64{{.5, -3.5, -2, 4}, {.5, -1, -.5, .5}, {1,0,0,0},},
			[]float64{0, 0, 1},
			[]float64{-1, 2, 0, 2},
			nil,
			nil,
			[]Bound{},
			[]float64{2, 2, 0, 0},
			0,
			4000,
			"Iteration limit reached.",
			false,
			false,
		},
		{
			// Basic feasible Cycling LP w/ Bland rule true
			// Example from Robert J. Vanderbei, 
			// Linear Programming: Foundations and Extensions Fourth Edition
			// page 26 (bottom of page) 
			// Example is maximization so need to use -c[] here
			// Case 2
			[][]float64{{.5, -3.5, -2, 4}, {.5, -1, -.5, .5}, {1,0,0,0},},
			[]float64{0, 0, 1},
			[]float64{-1, 2, 0, 2},
			nil,
			nil,
			[]Bound{},
			[]float64{2, 2, 0, 0},
			-1,
			7,
			"",
			true,
			false,
		},
		{
			// Basic feasible Cycling LP w/ dynamic bland rule
			// Example from Robert J. Vanderbei, 
			// Linear Programming: Foundations and Extensions Fourth Edition
			// page 26 (bottom of page)
			// Example is maximization so need to use -c[] 
			// Case 3
			[][]float64{{.5, -3.5, -2, 4}, {.5, -1, -.5, .5}, {1,0,0,0},},
			[]float64{0, 0, 1},
			[]float64{-1, 2, 0, 2},
			nil,
			nil,
			[]Bound{},
			[]float64{2, 2, 0, 0},
			-1,
			7,
			"",
			false,
			true,
		},
	}
	if testing.Short() {
		t.Skip("skipping Cycling tests (Bland pivot rules) in short mode")
	}

	//tol := 1.0E-12
	tol := 1.0e-7
	//bland := false
	maxiter := 4000 //4000
	//callback := meLPSimplexVerboseCallback
	//callback := LPSimplexVerboseCallback
	//callback := LPSimplexTerseCallback
	callback := Callbackfunc(nil)
	disp := true

	for i, elem := range tests {
		LPSimplexSetNewBehavior(NB_CMD_RESET, elem.dynamicBland) 
		start := time.Now()
		res := LPSimplex(elem.c, elem.a, elem.b, elem.ae, elem.be, elem.bounds, callback, disp, maxiter, tol, elem.useBland)
		//fmt.Printf("Res: %+v\n", res)
		//fmt.Printf("Case %d returned with success value of %v and objective value %v\n", i, res.Success, res.Fun)
		if !res.Success {
			if elem.errstr != res.Message {
				t.Errorf("TestLPsimplexCycle Case %d: failed with message %s\n", i, res.Message)
			}
		}
		elapsed := time.Since(start)
		degenCount := LPSimplexSetNewBehavior(NB_CMD_NOP, false) 
		fmt.Printf("	Degenerate Pivot Count is: %d\n", degenCount)
		fmt.Printf("Elapsed time is: %v\n", elapsed)
		if math.Abs(elem.opt-res.Fun) > tol {
			t.Errorf("TestLPsimplexCycle Case %d: Fun: %f but expected %f\n", i, res.Fun, elem.opt)
		}
		if elem.intr != res.Nitr {
			t.Errorf("TestLPsimplexCycle Case %d: Nitr: %d but expected %d\n", i, res.Nitr, elem.intr)
		}
		//e := reflect.DeepEqual(elem.x, res.X)
		e, str := doubleVectorEquals(elem.x, res.X)
		if e {
			t.Errorf("TestLPsimplexCycle Case %d: x: %v but expected: %v\n\tReason: %s", i, res.X, elem.x, str)
		}
	}
}
