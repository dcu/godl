package godl

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestEmbeddingGenerator(t *testing.T) {
	testCases := []struct {
		desc      string
		classes   int
		catDims   []int
		catIdxs   []int
		catEmbDim []int

		input               []float32
		inputShape          tensor.Shape
		expectedOutputShape tensor.Shape
		expectedOutput      []float32
		expectedGrad        []float32
		expectedCost        float32
	}{
		{
			// 0 1 2 3 4 5
			// cat idxs: 1 4
			desc:                "Example 1",
			classes:             5,
			catIdxs:             []int{1, 4},
			catDims:             []int{4, 4},
			catEmbDim:           []int{2, 2},
			input:               []float32{0, 1, 2, 1, 3},
			inputShape:          tensor.Shape{1, 5},
			expectedOutputShape: tensor.Shape{1, 7},
			expectedOutput:      []float32{0, 2, 3, 2, 1, 6, 7},
			expectedGrad:        []float32{0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285},
			expectedCost:        3,
		},
		{
			desc:                "Example 2",
			classes:             5,
			catIdxs:             []int{1, 4},
			catDims:             []int{4, 4},
			catEmbDim:           []int{2, 2},
			input:               []float32{0, 1, 2, 1, 3, 0, 1, 2, 1, 3},
			inputShape:          tensor.Shape{2, 5},
			expectedOutputShape: tensor.Shape{2, 7},
			expectedOutput:      []float32{0, 2, 3, 2, 1, 6, 7, 0, 2, 3, 2, 1, 6, 7},
			expectedGrad:        []float32{0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142},
			expectedCost:        3,
		},
	}
	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			tn := NewModel()
			embedder := EmbeddingGenerator(tn, tcase.classes, tcase.catDims, tcase.catIdxs, tcase.catEmbDim, EmbeddingOpts{
				WeightsInit: gorgonia.RangedFrom(0),
			})

			ts := tensor.New(
				tensor.WithShape(tcase.inputShape...),
				tensor.WithBacking(tcase.input),
			)

			input := gorgonia.NewTensor(tn.g, tensor.Float32, tcase.inputShape.Dims(), gorgonia.WithShape(tcase.inputShape...), gorgonia.WithValue(ts), gorgonia.WithName("input"))
			result, err := embedder(input)
			c.NoError(err)

			cost := gorgonia.Must(gorgonia.Mean(result.Output))
			_, err = gorgonia.Grad(cost, tn.Learnables()...)
			c.NoError(err)

			vm := gorgonia.NewTapeMachine(tn.g)
			c.NoError(vm.RunAll())

			c.Equal(tcase.expectedOutputShape, result.Shape())
			c.Equal(tcase.expectedOutput, result.Value().Data())

			yGrad, err := result.Output.Grad()
			c.NoError(err)

			c.Equal(tcase.expectedGrad, yGrad.Data())
			c.Equal(tcase.expectedCost, cost.Value().Data())
		})
	}
}
