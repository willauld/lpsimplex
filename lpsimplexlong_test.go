package lpsimplex

import (
	"fmt"
	"math"
	"testing"
	"time"
)

func TestRplan(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping Rplan big 1 in short mode")
	}
	var dynamicBland bool
	var big_1_intr_expected int

	for testno := 0; testno < 2; testno++ {
		if testno == 0 {
			dynamicBland = false
			big_1_intr_expected = 2898
		} else {
			dynamicBland = true
			big_1_intr_expected = 2104
		}
		A, b, c := GetModel_Big_1()
		//A, b, c := GetModelSmall_1()
		//c, A, b := binmodel.BinLoadModel("./RPlanModel.dat")
		fmt.Printf("Calling LPSimplex() for m:%d x n:%d model\n", len(A), len(A[0]))
		//tol := 1.0E-12
		tol := 1.0e-7
		bland := false
		maxiter := 4000 //4000
		//callback := meLPSimplexVerboseCallback
		//callback := LPSimplexVerboseCallback
		//callback := LPSimplexTerseCallback
		callback := Callbackfunc(nil)
		disp := true

		LPSimplexSetNewBehavior(NB_CMD_RESET, dynamicBland)
		start := time.Now()

		res := LPSimplex(c, A, b, nil, nil, nil, callback, disp, maxiter, tol, bland)
		elapsed := time.Since(start)
		degenCount := LPSimplexSetNewBehavior(NB_CMD_NOP, false)
		fmt.Printf("	Degenerate Pivot Count is: %d\n", degenCount)
		fmt.Printf("Elapsed time is: %v\n", elapsed)
		if res.Success != true {
			t.Errorf("big_1 returned Success: %v and message: %s\n", res.Success, res.Message)
		}
		big_1_expected := -1.3137417053996125e+07
		if math.Abs(res.Fun-big_1_expected) > tol {
			t.Errorf("big_1 returned Fun: %f but expected %f\n", res.Fun, big_1_expected)
		}
		// set above: big_1_intr_expected := 2104 //2405 // 2898, dynamicBland true, with 200mdegen, false
		if res.Nitr != big_1_intr_expected {
			t.Errorf("big_1 returned interations: %d but expected %d\n", res.Nitr, big_1_intr_expected)
		}
		ms := 1000000
		s := 1000 * ms
		big_1_time := time.Duration(55 * s)
		if elapsed > big_1_time {
			t.Errorf("big_1 time: %s but expected it to be less than %s\n", elapsed, big_1_time)
		}
		//fmt.Printf("\n***** LPSimplex() took %s *****\n\n", elapsed)
		//fmt.Printf("Res: %+v\n", res)
	}
}
