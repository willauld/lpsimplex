package lpsimplex

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/willauld/temp/topme/load_model/binmodel"
)

func TestRplan(t *testing.T) {
	//A, b, c := GetModel()
	c, A, b := binmodel.BinLoadModel("./load_model/RPlanModel.dat")
	fmt.Printf("Calling linprog_simplex() for m:%d x n:%d model\n", len(A), len(A[0]))
	tol := 1.0E-12
	bland := false
	maxiter := 2000
	//callback := LPSimplexVerboseCallback
	//callback := LPSimplexTerseCallback
	callback := Callbackfunc(nil)
	disp := true

	start := time.Now()

	res := LPSimplex(c, A, b, nil, nil, nil, callback, disp, maxiter, tol, bland)
	elapsed := time.Since(start)
	fmt.Printf("\n***** linprog_simplex() took %s *****\n\n", elapsed)
	fmt.Printf("Res: %+v\n", res)
}

func TestLinprog(t *testing.T) {
	tests := []struct {
		a      [][]float64
		b      []float64
		c      []float64
		bounds []Bound
		x      []float64
		opt    float64
		errstr string
	}{
		// Basic feasible LP
		{[][]float64{{-1, 2, 1, 0}, {3, 1, 0, 1}},
			[]float64{4, 9},
			[]float64{-1, -2, 0, 0},
			[]Bound{},
			[]float64{2, 2, 0, 0},
			-8,
			"",
		},
		{[][]float64{{-3, 1}, {1, 2}},
			[]float64{6, 4},
			[]float64{-1, 4},
			[]Bound{},
			[]float64{0, 0},
			-22,
			"",
		},
		{[][]float64{{-3, 1}, {1, 2}},
			[]float64{6, 4},
			[]float64{-1, 4},
			[]Bound{{math.Inf(-1), math.Inf(1)}, {-3, math.Inf(1)}},
			[]float64{0, 0},
			-22,
			"",
		},
	}

	tol := float64(0)
	//tol=1.0E-12
	bland := false
	maxiter := 10
	//bounds := []Bound(nil)
	//callback := LPSimplexVerboseCallback
	//callback := LPSimplexTerseCallback
	callback := Callbackfunc(nil)
	disp := true

	for i, elem := range tests {
		//x, fun, nit, status, slack, message, successful := linprog_simplex(elem.c, elem.a, elem.b, nil, nil, elem.bounds, callback, disp, maxiter, tol, bland)
		res := LPSimplex(elem.c, elem.a, elem.b, nil, nil, elem.bounds, callback, disp, maxiter, tol, bland)
		fmt.Printf("Res: %+v\n", res)
		/*
			fmt.Printf("linprob_simplex says: %s\n", message)
			fmt.Printf("\ninterations: %d\n", nit)
			if len(x) != len(elem.c) {
				if successful == true {
					fmt.Printf("len(x:%d) != len(c:%d) but should be\n", len(x), len(elem.c))
				} else {
					if x[0] != math.NaN() { // This does not work correctly TODO
						fmt.Printf("Simplex failed: BUT x[0] != NaN it is: %v\n", x[0])
					}
				}
			}
			fmt.Printf("Status: %d\n", status)
			if successful && len(slack) != len(elem.a) {
				fmt.Printf("Slack is not the currect lenth. Should be %d but is %d\n", len(elem.a), len(slack))
			}
			fmt.Printf("Case %d returned with success value of %v and objective value %f\n", i, successful, fun)
		*/
		fmt.Printf("Case %d returned with success value of %v and objective value %f\n", i, res.success, res.fun)
	}
}

func TestPrintT(t *testing.T) {
	tests := []struct {
		a [][]float64
	}{
		{[][]float64{{-1, 2, 1, 0}, {3, 1, 0, 1}}},
	}

	for i, elem := range tests {
		fmt.Printf("Case %d: \n", i)
		err := TersPrintMatrix(elem.a)
		if err != nil {
			fmt.Printf("Error - %s\n", err)
		}
		// TODO: redirect IO and compare with expected result
	}
}

func TestCountNegEntries(t *testing.T) {
	tests := []struct {
		a      []float64
		expect int
	}{
		{[]float64{1, 2, 3, 4, 5, 6}, 0},
		{[]float64{1, 2, 3, -4, 5, 6}, 1},
		{[]float64{}, 0},
		{nil, 0},
		{[]float64{-1, -2, -3, -4, -5, -6}, 6},
	}
	for i, elem := range tests {
		r := countNegEntries(elem.a)
		if r != elem.expect {
			t.Errorf("Case %d failed: countNegEntries() returned \"%d\" but expected \"%d\"\n", i, r, elem.expect)
		}
	}
}

func TestCheckRectangle(t *testing.T) {
	tests := []struct {
		a          [][]float64
		rows, cols int
		errstr     string
	}{
		{[][]float64{{1, 2}, {3, 4}, {5, 6}}, 3, 2, ""},
		{[][]float64{{1}, {2}, {3}, {4}}, 0, 0, "Invalid input, must be two-dimensional"},
		{[][]float64{}, 0, 0, "Invalid input, must be two-dimensional"},
		{[][]float64{{1, 2}, {3, 4, 10}, {5, 6}}, 0, 0, "Invalid input, all rows must have the same length"},
		{nil, 0, 0, ""},
	}
	for i, elem := range tests {
		rows, cols, err := checkRectangle(elem.a)
		if err != nil {
			if err.Error() != elem.errstr {
				t.Errorf("Case %d failed: checkRectangle() returned \"%v\" but expected \"%s\"\n", i, err, elem.errstr)
			}
		}
		if err == nil && elem.errstr != "" {
			t.Errorf("Case %d failed: checkRectangle() returned nil error but expected error \"%s\"\n", i, elem.errstr)
		}
		if elem.rows != rows {
			t.Errorf("Case %d failed: checkRectangle() returned rows: %d but expected %d\n", i, rows, elem.rows)
		}
		if elem.cols != cols {
			t.Errorf("Case %d failed: checkRectangle() returned cols: %d but expected %d\n", i, cols, elem.cols)
		}
	}
}

/*
func TestGetQuantity(t *testing.T) {
	testCases := []struct {
		line  string
		quant int
	}{
		{"- 1 other stuff", 1},
		{" some stuff 5 other stuff", 5},
		{"- 10 other stuff", 10},
		{"- 162 other stuff", 162},
		{"-1 other stuff", 1},
	}
	for _, testCase := range testCases {
		quantity := getQuantity(testCase.line)
		if quantity != testCase.quant {
			t.Errorf("getQuantity(%s) returned %d but expected %d\n", testCase.line, quantity, testCase.quant)
		}
	}
}

func TestGetTitle(t *testing.T) {
	tstCases := []struct {
		line     string
		expected string
	}{
		{"- 56 A title 76 -- not this $ 6.43", "A title 76"},
		{"- A title -- not this $ 6.43", "A title"},
		{"- A title not this $ 6.43", "A title not this"},
		{"- A title ", "A title"},
		{"A title ", "A title"},
	}
	for _, tstCase := range tstCases {
		title := getTitle(tstCase.line)
		if title != tstCase.expected {
			t.Errorf("getTitle(%s) returned [%s] but [%s] was expected\n", tstCase.line, title, tstCase.expected)
		}
	}
}

func TestGetPrice(t *testing.T) {
	tstCases := []struct {
		line     string
		expected float32
	}{
		{"$6.50", 6.50},
		{" - 23 This is the title -- other stuff $ 6.50  ", 6.50},
	}
	for _, tstCase := range tstCases {
		price := getPrice(tstCase.line)
		if price != tstCase.expected {
			t.Errorf("getPrice(%s) returned [%6.2f] expected [%6.2f]\n", tstCase.line, price, tstCase.expected)
		}
	}
}
*/
