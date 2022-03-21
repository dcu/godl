package table

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestRandValueIn(t *testing.T) {
	testCases := []struct {
		valsAndProbs map[string]float64
	}{
		{
			valsAndProbs: map[string]float64{
				"large":  0.8,
				"medium": 0.1,
				"small":  0.1,
			},
		},
	}
	for i, tC := range testCases {
		t.Run(fmt.Sprintf("Example #%d", i+1), func(t *testing.T) {
			c := require.New(t)

			rv := RandValueIn(tC.valsAndProbs)
			counters := map[string]int{}
			n := 100

			for i := 0; i < n; i++ {
				counters[rv()]++
			}

			t.Logf("counters: %#v", counters)

			for val, prob := range tC.valsAndProbs {
				expected := prob * float64(n)

				c.InDelta(expected, counters[val], 5)
			}
		})
	}
}

func TestReadCSV(t *testing.T) {
	testCases := []struct {
		csvPath            string
		SumColumn          int
		ExpectedSum        float64
		SliceAt            int
		ExpectedSliceValue []float32

		TargetColumns      []int
		ExpectedCatIndexes []int
		ExpectedCatDims    []int
	}{
		{
			csvPath:            "fixtures/fruits.csv",
			SliceAt:            1,
			ExpectedSliceValue: []float32{2, 200},
			SumColumn:          1,
			ExpectedSum:        450,
			TargetColumns:      []int{2},
			ExpectedCatIndexes: []int{0},
			ExpectedCatDims:    []int{3},
		},
		{
			csvPath:            "fixtures/census.csv",
			SliceAt:            11,
			ExpectedSliceValue: []float32{30, 2, 141297, 5, 13, 1, 7, 0, 1, 1, 0, 0, 40, 2},
			SumColumn:          4,
			ExpectedSum:        212,
			TargetColumns:      []int{14},
			ExpectedCatIndexes: []int{1, 3, 5, 6, 7, 8, 9, 13},
			ExpectedCatDims:    []int{3, 9, 4, 10, 5, 4, 2, 6},
		},
	}
	for i, tC := range testCases {
		t.Run(fmt.Sprintf("Example #%d (%s)", i+1, tC.csvPath), func(t *testing.T) {
			c := require.New(t)
			table, err := ReadCSV(tC.csvPath)
			c.NoError(err)

			t.Logf("rows: %#v", table.Rows)

			table.EachColumn(func(columnName string, v Cell) {
				if columnName == " 100" && v.Dtype == tensor.Int64 {
					c.Equal(100, v.V)
				}
			})

			sum := float32(0.0)
			table.EachRow(func(row Row) {
				v := row.Cells[tC.SumColumn]
				if v.Dtype == tensor.Int {
					sum += float32(row.Cells[tC.SumColumn].Int())
				} else if v.Dtype == tensor.Float32 {
					sum += row.Cells[tC.SumColumn].Float32()
				}
			})

			c.Equal(tC.ExpectedSum, sum)

			t.Logf("classes: %v", table.ClassesByColumn)

			x, y := table.ToTensors(ToTensorOpts{
				TargetColumns: tC.TargetColumns,
			})
			s, err := x.Slice(tensor.S(tC.SliceAt))
			c.NoError(err)

			t.Logf("x:\n%#v %v", x, x.Shape())

			if y != nil {
				t.Logf("y:\n%#v %v", y, y.Shape())
			}

			c.Equal(tC.ExpectedSliceValue, s.Data())

			idx, dims := table.CategoricalColumns(tC.TargetColumns...)
			t.Logf("%v %v", idx, dims)

			c.Equal(tC.ExpectedCatIndexes, idx)
			c.Equal(tC.ExpectedCatDims, dims)
		})
	}
}
