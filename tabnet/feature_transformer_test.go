package tabnet

import (
	"testing"

	"github.com/dcu/godl"
	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestFeatureTransformer(t *testing.T) {
	testCases := []struct {
		desc              string
		input             tensor.Tensor
		weight            []float32
		vbs               int
		independentBlocks int
		sharedBlocks      int
		output            int
		expectedShape     tensor.Shape
		expectedErr       string
		expectedOutput    []float32
		expectedGrad      []float32
		expectedCost      float32
	}{
		{
			desc: "Example1",
			input: tensor.New(
				tensor.WithShape(6, 2),
				tensor.WithBacking([]float32{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.4}),
			),
			weight:            []float32{-2.1376, -1.7072, 1.5896, 1.3657, 0.2603, -0.4051, -0.8271, 1.0830, 0.2617, -0.2792, -1.2426, 0.8678, 1.0771, -0.1787, -0.7482, 0.9506, -0.0861, -0.3015, -1.0695, -0.0246, -0.7007, 0.0354, -0.2400, -0.2516, -0.3165, 2.0425, 0.6425, 0.5848, -1.9183, 0.0099, -0.8387, -1.5346},
			vbs:               2,
			output:            8 + 8,
			independentBlocks: 5,
			sharedBlocks:      5,
			expectedShape:     tensor.Shape{6, 2},
			expectedOutput:    []float32{-0.24453057, 2.446192, 0.664702, -6.64944, -0.2445306, 2.446192, 0.66470236, -6.6494403, -0.2445307, 2.4461923, 0.6647016, -6.64944},
			expectedGrad:      []float32{0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333},
			expectedCost:      -0.94576913,
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)
			tn := godl.NewModel()
			g := tn.TrainGraph()

			input := gorgonia.NewTensor(g, tensor.Float32, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithName("x"), gorgonia.WithValue(tcase.input))

			fcWeight := gorgonia.NewTensor(g, tensor.Float32, 2, gorgonia.WithShape(input.Shape()[1], tcase.output), gorgonia.WithValue(
				tensor.New(
					tensor.WithShape(input.Shape()[1], tcase.output),
					tensor.WithBacking(tcase.weight),
				),
			), gorgonia.WithName("fcWeight"))

			shared := make([]*godl.LinearModule, tcase.sharedBlocks)
			fcInput := input.Shape()[1]
			fcOutput := 2 * tcase.output
			for i := 0; i < tcase.sharedBlocks; i++ {
				shared[i] = godl.Linear(tn, godl.LinearOpts{
					OutputDimension: fcOutput, // double the size so we can take half and half
					WeightsInit:     gorgonia.RangedFromWithStep(-0.1, 0.01),
					InputDimension:  fcInput,
				})

				fcInput = tcase.output
			}

			result := FeatureTransformer(tn, FeatureTransformerOpts{
				VirtualBatchSize:  tcase.vbs,
				InputDimension:    input.Shape()[1],
				OutputDimension:   tcase.output,
				Shared:            shared,
				IndependentBlocks: tcase.independentBlocks,
				WeightsInit:       initDummyWeights,
				Momentum:          0.02,
			}).Forward(input)

			y := result[0]

			wT := gorgonia.Must(gorgonia.Transpose(fcWeight, 1, 0))
			y = gorgonia.Must(gorgonia.Mul(y, wT))

			cost := gorgonia.Must(gorgonia.Mean(y))
			_, err := gorgonia.Grad(cost, input)
			c.NoError(err)

			vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(tn.Learnables()...))
			c.NoError(vm.RunAll())

			tn.PrintWatchables()

			t.Logf("feat output: %v", y.Value())
			t.Logf("y: %v", y.Value())
			t.Logf("dx: %v", input.Deriv().Value())

			c.Equal(tcase.expectedShape, y.Shape())
			c.InDeltaSlice(tcase.expectedOutput, y.Value().Data().([]float32), 1e-5, "%#v expected. Got: %#v", tcase.expectedOutput, y.Value().Data())

			yGrad, err := y.Grad()
			c.NoError(err)

			c.Equal(tcase.expectedGrad, yGrad.Data())
			c.InDelta(tcase.expectedCost, cost.Value().Data(), 1e-5)
		})
	}
}
