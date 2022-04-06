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
		priors            []float32
		vbs               int
		output            int
		expectedShape     tensor.Shape
		expectedErr       string
		expectedOutput    []float32
		expectedGrad      []float32
		expectedCost      float32
		expectedInputGrad []float32
	}{
		{
			desc: "Example 1",
			input: tensor.New(
				tensor.WithShape(6, 2),
				tensor.WithBacking([]float32{0.1, -0.5, 0.3, 0.9, 0.04, -0.3, 0.01, 0.09, -0.1, 0.9, 0.7, 0.04}),
			),
			priors:            []float32{-1.0143, 0.9077, 0.8760, -2.8345, 0.9163, -1.5155, -0.8302, 0.5957, -0.9591, 0.4161, -0.2541, 0.6725},
			vbs:               2,
			output:            2,
			expectedShape:     tensor.Shape{6, 2},
			expectedOutput:    []float32{-0.05, 0.009999998, -0.05, 0.009999998, -0.020000001, 0.04, -0.020000001, 0.04, -0.020000001, 0.04, -0.04882242, 0.01117758},
			expectedGrad:      []float32{0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333},
			expectedCost:      -0.0048037386,
			expectedInputGrad: []float32{0, 0, 0, 0, 0, 0, 0, 0, -0.00042192982004980896, -0.00042192982004980896, 0.00042192982004980896, 0.00042192982004980896},
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			tn := godl.NewModel()
			g := tn.TrainGraph()

			input := gorgonia.NewTensor(g, tensor.Float32, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithName("input"), gorgonia.WithValue(tcase.input))
			priors := gorgonia.NewTensor(g, tensor.Float32, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithValue(
				tensor.New(
					tensor.WithShape(input.Shape()...),
					tensor.WithBacking(tcase.priors),
				),
			),
				gorgonia.WithName("priors"),
			)
			result := AttentiveTransformer(tn, AttentiveTransformerOpts{
				VirtualBatchSize: tcase.vbs,
				InputDimension:   input.Shape()[1],
				OutputDimension:  tcase.output,
				WeightsInit:      initDummyWeights,
			}).Forward(input, priors)

			fcWeight := gorgonia.NewTensor(g, tensor.Float32, 2, gorgonia.WithShape(input.Shape()[1], tcase.output), gorgonia.WithInit(gorgonia.RangedFromWithStep(-0.05, 0.03)), gorgonia.WithName("fcWeight"))

			y := result[0]
			wT := gorgonia.Must(gorgonia.Transpose(fcWeight, 1, 0))
			y = gorgonia.Must(gorgonia.Mul(y, wT))

			cost := gorgonia.Must(gorgonia.Mean(y))
			_, err := gorgonia.Grad(cost, input)
			c.NoError(err)

			vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(append(tn.Learnables(), fcWeight)...))
			c.NoError(vm.RunAll())

			tn.PrintWatchables()

			t.Logf("input: %v", input.Value())
			t.Logf("priors: %v", priors.Value())
			t.Logf("dx: %v", input.Deriv().Value())
			t.Logf("att output: %v", y.Value())

			c.Equal(tcase.expectedShape, y.Shape())
			c.Equal(tcase.expectedOutput, y.Value().Data().([]float32))

			yGrad, err := y.Grad()
			c.NoError(err)

			c.Equal(tcase.expectedGrad, yGrad.Data())
			c.Equal(tcase.expectedCost, cost.Value().Data())

			c.InDeltaSlice(tcase.expectedInputGrad, input.Deriv().Value().Data(), 1e-5, "actual: %#v", input.Deriv().Value().Data())
		})
	}
}
