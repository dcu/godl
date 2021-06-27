package godl

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestGBN(t *testing.T) {
	testCases := []struct {
		desc           string
		input          tensor.Tensor
		vbs            int
		expectedShape  tensor.Shape
		expectedErr    string
		expectedOutput []float32
		expectedGrad   []float32
		expectedCost   float32
	}{
		{
			desc: "Example 1",
			input: tensor.New(
				tensor.WithShape(10, 1),
				tensor.WithBacking([]float32{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4}),
			),
			vbs:            5,
			expectedShape:  tensor.Shape{10, 1},
			expectedOutput: []float32{-1.41421, -0.70710504, 0, 0.707105, 1.41421, -1.4142102, -0.7071051, 0, 0.70710474, 1.4142098},
			expectedGrad:   []float32{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
			expectedCost:   -9.536743e-08,
		},
		{
			desc: "Example 2",
			input: tensor.New(
				tensor.WithShape(5, 2),
				tensor.WithBacking([]float32{0.4, -1.4, 2.4, -3.4, 4.4, -5.4, 6.4, -7.4, 8.4, -9.4}),
			),
			vbs:            5,
			expectedShape:  tensor.Shape{5, 2},
			expectedOutput: []float32{-1.4142127, 1.4142127, -0.70710635, 0.70710635, 0, 0, 0.70710635, -0.70710635, 1.4142126, -1.4142126},
			expectedGrad:   []float32{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
			expectedCost:   0,
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			tn := NewModel()
			tn.Training = true
			g := tn.ExprGraph()

			input := gorgonia.NewTensor(g, tensor.Float32, 2, gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithName("GBNInput"), gorgonia.WithValue(tcase.input))

			y, err := GBN(tn, GBNOpts{
				VirtualBatchSize: tcase.vbs,
				OutputDimension:  tcase.input.Shape()[1],
			})(input)
			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			}
			c.NoError(err)

			cost := gorgonia.Must(gorgonia.Mean(y.Output))
			_, err = gorgonia.Grad(cost, input)
			c.NoError(err)

			c.Equal(tcase.expectedShape, y.Shape())

			vm := gorgonia.NewTapeMachine(tn.g,
				gorgonia.WithLogger(testLogger),
				gorgonia.BindDualValues(tn.learnables...),
				gorgonia.WithValueFmt("%+v"),
				gorgonia.WithWatchlist(),
			)
			c.NoError(vm.RunAll())

			yGrad, err := y.Output.Grad()
			c.NoError(err)

			c.Equal(tcase.expectedOutput, y.Value().Data().([]float32))
			c.Equal(tcase.expectedCost, cost.Value().Data())
			c.Equal(tcase.expectedGrad, yGrad.Data())
		})
	}
}
