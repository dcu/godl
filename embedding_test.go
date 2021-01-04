package tabnet

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestEmbedding(t *testing.T) {
	testCases := []struct {
		desc           string
		classes        int
		dim            int
		input          []float64
		inputShape     tensor.Shape
		expectedOutput []float64
	}{
		{
			desc:           "Example 1",
			input:          []float64{1, 2},
			inputShape:     tensor.Shape{2},
			classes:        4,
			dim:            2,
			expectedOutput: []float64{2, 3, 4, 5},
		},
		{
			desc:           "Example 2",
			input:          []float64{1, 2, 2, 0},
			inputShape:     tensor.Shape{4},
			classes:        4,
			dim:            2,
			expectedOutput: []float64{2, 3, 4, 5, 4, 5, 0, 1},
		},
		{
			desc:           "Example 3",
			input:          []float64{2},
			inputShape:     tensor.Shape{1, 1, 1},
			classes:        4,
			dim:            2,
			expectedOutput: []float64{4, 5},
		},
		{
			desc:           "Example 4",
			input:          []float64{0, 3, 2, 1},
			inputShape:     tensor.Shape{2, 1, 2},
			classes:        4,
			dim:            2,
			expectedOutput: []float64{0, 1, 6, 7, 4, 5, 2, 3},
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			tn := NewModel()
			emb := tn.Embedding(tcase.classes, tcase.dim, EmbeddingOpts{
				WeightsInit: gorgonia.RangedFrom(0),
			})

			ts := tensor.New(tensor.WithShape(tcase.inputShape...), tensor.WithBacking(tcase.input))

			selector := gorgonia.NewTensor(tn.g, tensor.Float64, tcase.inputShape.Dims(), gorgonia.WithShape(ts.Shape()...), gorgonia.WithValue(ts), gorgonia.WithName("selector"))
			output, err := emb(selector)
			c.NoError(err)

			vm := gorgonia.NewTapeMachine(tn.g)
			c.NoError(vm.RunAll())

			c.Equal(append(tcase.inputShape, tcase.dim), output.Shape())
			c.Equal(tcase.expectedOutput, output.Value().Data())
		})
	}
}
