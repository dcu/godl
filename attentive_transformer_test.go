package tabnet

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestAttentiveTransformer(t *testing.T) {
	testCases := []struct {
		desc           string
		input          tensor.Tensor
		vbs            int
		output         int
		expectedShape  tensor.Shape
		expectedErr    string
		expectedOutput []float32
	}{
		{
			desc: "Example 1",
			input: tensor.New(
				tensor.WithShape(6, 2),
				tensor.WithBacking([]float32{0.1, -0.5, 0.3, 0.9, 0.04, -0.3, 0.01, 0.09, -0.1, 0.9, 0.7, 0.04}),
			),
			vbs:            2,
			output:         2,
			expectedShape:  tensor.Shape{6, 2},
			expectedOutput: []float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			g := gorgonia.NewGraph()
			tn := &Model{g: g}

			input := gorgonia.NewTensor(g, tensor.Float32, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithName("input"), gorgonia.WithValue(tcase.input))
			priors := gorgonia.NewTensor(g, tensor.Float32, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithInit(gorgonia.Ones()))
			x, _, err := tn.AttentiveTransformer(AttentiveTransformerOpts{
				VirtualBatchSize: tcase.vbs,
				InputDimension:   input.Shape()[1],
				OutputDimension:  tcase.output,
				WeightsInit:      initDummyWeights,
			})(input, priors)

			if tcase.expectedErr != "" {
				c.Error(err)
				c.Equal(tcase.expectedErr, err.Error())

				return
			} else {
				c.NoError(err)
			}

			vm := gorgonia.NewTapeMachine(g)
			c.NoError(vm.RunAll())

			c.Equal(tcase.expectedShape, x.Shape())
			c.Equal(tcase.expectedOutput, x.Value().Data().([]float32))
		})
	}
}
