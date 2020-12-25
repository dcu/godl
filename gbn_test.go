package tabnet

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestGBN(t *testing.T) {
	testCases := []struct {
		desc           string
		input          tensor.Tensor
		vbs            int
		expectedShape  tensor.Shape
		expectedErr    string
		expectedOutput []float64
	}{
		{
			desc: "Example 1",
			input: tensor.New(
				tensor.WithShape(10, 1),
				tensor.WithBacking([]float64{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4}),
			),
			vbs:            5,
			expectedShape:  tensor.Shape{10, 1},
			expectedOutput: []float64{-1.4142100268524476, -0.7071050134262239, -3.140177066934696e-16, 0.7071050134262233, 1.4142100268524473, -1.4142100268524478, -0.7071050134262242, -6.280354133869392e-16, 0.707105013426223, 1.4142100268524467},
		},
		{
			desc: "Example 2",
			input: tensor.New(
				tensor.WithShape(5, 2),
				tensor.WithBacking([]float64{0.4, -1.4, 2.4, -3.4, 4.4, -5.4, 6.4, -7.4, 8.4, -9.4}),
			),
			vbs:            5,
			expectedShape:  tensor.Shape{5, 2},
			expectedOutput: []float64{-1.4142126784904472, 1.4142126784904472, -0.7071063392452237, 0.7071063392452237, 0, 0, 0.7071063392452236, -0.7071063392452236, 1.4142126784904472, -1.4142126784904472},
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			g := NewGraph()
			tn := &Model{g: g}
			input := NewTensor(g, tensor.Float64, 2, WithShape(tcase.input.Shape()...), WithName("GBNInput"), WithValue(tcase.input))

			y, err := tn.GBN(GBNOpts{
				VirtualBatchSize: tcase.vbs,
			})(input)

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			}

			c.NoError(err)
			c.Equal(tcase.expectedShape, y.Shape())

			vm := NewTapeMachine(tn.g,
				gorgonia.WithLogger(testLogger),
				gorgonia.BindDualValues(tn.learnables...),
				gorgonia.WithValueFmt("%+v"),
				gorgonia.WithWatchlist(),
			)
			c.NoError(vm.RunAll())

			c.Equal(tcase.expectedOutput, y.Value().Data().([]float64))
		})
	}
}
