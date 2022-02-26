package tabnet

import (
	"testing"

	"github.com/dcu/godl"
	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestAttentiveTransformer(t *testing.T) {
	testCases := []struct {
		desc              string
		input             tensor.Tensor
		priors            []float64
		vbs               int
		output            int
		expectedShape     tensor.Shape
		expectedErr       string
		expectedOutput    []float64
		expectedGrad      []float64
		expectedCost      float64
		expectedInputGrad []float64
	}{
		{
			desc: "Example 1",
			input: tensor.New(
				tensor.WithShape(6, 2),
				tensor.WithBacking([]float64{0.1, -0.5, 0.3, 0.9, 0.04, -0.3, 0.01, 0.09, -0.1, 0.9, 0.7, 0.04}),
			),
			priors:            []float64{-1.0143, 0.9077, 0.8760, -2.8345, 0.9163, -1.5155, -0.8302, 0.5957, -0.9591, 0.4161, -0.2541, 0.6725},
			vbs:               2,
			output:            2,
			expectedShape:     tensor.Shape{6, 2},
			expectedOutput:    []float64{-0.05, 0.009999999999999995, -0.05, 0.009999999999999995, -0.020000000000000004, 0.039999999999999994, -0.020000000000000004, 0.039999999999999994, -0.020000000000000004, 0.039999999999999994, -0.04882242090483175, 0.011177579095168255},
			expectedGrad:      []float64{0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333},
			expectedCost:      -0.004803736817471961,
			expectedInputGrad: []float64{0, 0, 0, 0, 0, 0, 0, 0, -0.00042192982004980896, -0.00042192982004980896, 0.00042192982004980896, 0.00042192982004980896},
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			tn := godl.NewModel()
			g := tn.ExprGraph()

			input := gorgonia.NewTensor(g, tensor.Float64, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithName("input"), gorgonia.WithValue(tcase.input))
			priors := gorgonia.NewTensor(g, tensor.Float64, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithValue(
				tensor.New(
					tensor.WithShape(input.Shape()...),
					tensor.WithBacking(tcase.priors),
				),
			),
				gorgonia.WithName("priors"),
			)
			result, err := AttentiveTransformer(tn, AttentiveTransformerOpts{
				VirtualBatchSize: tcase.vbs,
				InputDimension:   input.Shape()[1],
				OutputDimension:  tcase.output,
				WeightsInit:      initDummyWeights,
			})(input, priors)

			fcWeight := gorgonia.NewTensor(g, tensor.Float64, 2, gorgonia.WithShape(input.Shape()[1], tcase.output), gorgonia.WithInit(gorgonia.RangedFromWithStep(-0.05, 0.03)), gorgonia.WithName("fcWeight"))

			y := result.Output
			wT := gorgonia.Must(gorgonia.Transpose(fcWeight, 1, 0))
			y = gorgonia.Must(gorgonia.Mul(y, wT))

			if tcase.expectedErr != "" {
				c.Error(err)
				c.Equal(tcase.expectedErr, err.Error())

				return
			} else {
				c.NoError(err)
			}

			cost := gorgonia.Must(gorgonia.Mean(y))
			_, err = gorgonia.Grad(cost, input)
			c.NoError(err)

			vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(append(tn.Learnables(), fcWeight)...))
			c.NoError(vm.RunAll())

			tn.PrintWatchables()

			t.Logf("input: %v", input.Value())
			t.Logf("priors: %v", priors.Value())
			t.Logf("dx: %v", input.Deriv().Value())
			t.Logf("att output: %v", result.Output.Value())

			c.Equal(tcase.expectedShape, y.Shape())
			c.Equal(tcase.expectedOutput, y.Value().Data().([]float64))

			yGrad, err := y.Grad()
			c.NoError(err)

			c.Equal(tcase.expectedGrad, yGrad.Data())
			c.Equal(tcase.expectedCost, cost.Value().Data())

			c.InDeltaSlice(tcase.expectedInputGrad, input.Deriv().Value().Data(), 1e-5, "actual: %#v", input.Deriv().Value().Data())
		})
	}
}
