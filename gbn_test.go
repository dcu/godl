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
		expectedOutput []float64
		expectedGrad   []float64
		expectedCost   float64
	}{
		// {
		// 	desc: "Example 1",
		// 	input: tensor.New(
		// 		tensor.WithShape(10, 1),
		// 		tensor.WithBacking([]float64{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4}),
		// 	),
		// 	vbs:            5,
		// 	expectedShape:  tensor.Shape{10, 1},
		// 	expectedOutput: []float64{-1.4142100268524473, -0.7071050134262237, -1.8394620353687656e-17, 0.7071050134262237, 1.4142100268524476, -1.4142100268524476, -0.7071050134262239, -2.5948842597279634e-16, 0.7071050134262234, 1.4142100268524471},
		// 	expectedGrad:   []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
		// 	expectedCost:   -1.3322676295501878e-16,
		// },
		{
			desc: "Example 2",
			input: tensor.New(
				tensor.WithShape(5, 2),
				tensor.WithBacking([]float64{0.4, -1.4, 2.4, -3.4, 4.4, -5.4, 6.4, -7.4, 8.4, -9.4}),
			),
			vbs:            5,
			expectedShape:  tensor.Shape{5, 2},
			expectedOutput: []float64{-1.4142126784904472, 1.4142126784904474, -0.7071063392452236, 0.7071063392452238, 1.0340285769764954e-16, 6.313059599612395e-17, 0.7071063392452237, -0.7071063392452235, 1.4142126784904472, -1.4142126784904472},
			expectedGrad:   []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
			expectedCost:   8.881784197001253e-17,
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			tn := NewModel()
			g := tn.ExprGraph()

			input := gorgonia.NewTensor(g, tensor.Float64, 2, gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithName("GBNInput"), gorgonia.WithValue(tcase.input))

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
			_, err = gorgonia.Grad(cost, append(tn.Learnables(), input)...)
			c.NoError(err)

			c.Equal(tcase.expectedShape, y.Shape())

			vm := gorgonia.NewTapeMachine(tn.g,
				gorgonia.WithLogger(testLogger),
				gorgonia.BindDualValues(tn.learnables...),
				gorgonia.WithValueFmt("%+v"),
				gorgonia.WithWatchlist(),
			)
			c.NoError(vm.RunAll())

			t.Logf("dx: %v", input.Deriv().Value())

			yGrad, err := y.Output.Grad()
			c.NoError(err)

			c.InDeltaSlice(tcase.expectedOutput, y.Value().Data().([]float64), 1e-5, "actual: %#v", y.Value().Data())
			c.InDelta(tcase.expectedCost, cost.Value().Data(), 1e-5, "actual: %#v", cost.Value().Data())
			c.InDeltaSlice(tcase.expectedGrad, yGrad.Data(), 1e-5, "actual: %#v", yGrad.Data())
		})
	}
}
