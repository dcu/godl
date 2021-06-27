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
			vbs:               2,
			output:            2,
			independentBlocks: 5,
			sharedBlocks:      0,
			expectedShape:     tensor.Shape{6, 2},
			expectedOutput:    []float32{-0.46379328, -0.28701657, 1.8771948, 2.0539715, 0.2433135, 0.4200901, 2.5843015, 2.7610784, 0.9504202, 1.1271969, 3.291408, 3.468185},
			expectedGrad:      []float32{0.083333336, 0.083333336, 0.083333336, 0.083333336, 0.083333336, 0.083333336, 0.083333336, 0.083333336, 0.083333336, 0.083333336, 0.083333336, 0.083333336},
			expectedCost:      1.502196,
		},
		{
			desc: "Example2",
			input: tensor.New(
				tensor.WithShape(6, 2),
				tensor.WithBacking([]float32{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.4}),
			),
			vbs:               2,
			output:            16,
			independentBlocks: 5,
			sharedBlocks:      5,
			expectedShape:     tensor.Shape{6, 16},
			expectedOutput:    []float32{-0.63247323, -0.63247323, -0.63247323, -0.63247323, -0.63247323, -0.63247323, -0.63247323, -0.63247323, -0.63247323, -0.63247323, -0.63247323, -0.63247323, -0.63247323, -0.63247323, -0.63247323, -0.63247323, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, -0.6324731, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404, 1.7192404},
			expectedGrad:      []float32{0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667, 0.010416667},
			expectedCost:      0.5433839,
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			tn := godl.NewModel()
			tn.Training = true

			g := tn.ExprGraph()

			input := gorgonia.NewTensor(g, tensor.Float32, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithName("input"+tcase.desc), gorgonia.WithValue(tcase.input))

			shared := make([]godl.Layer, tcase.sharedBlocks)
			fcInput := input.Shape()[1]
			fcOutput := 2 * (8 + 8)
			for i := 0; i < tcase.sharedBlocks; i++ {
				shared[i] = godl.FC(tn, godl.FCOpts{
					OutputDimension: fcOutput, // double the size so we can take half and half
					WeightsInit:     initDummyWeights,
					InputDimension:  fcInput,
				})

				fcInput = 8 + 8
			}

			y, err := FeatureTransformer(tn, FeatureTransformerOpts{
				VirtualBatchSize:  tcase.vbs,
				InputDimension:    input.Shape()[1],
				OutputDimension:   tcase.output,
				Shared:            shared,
				IndependentBlocks: tcase.independentBlocks,
				WeightsInit:       initDummyWeights,
			})(input)

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			} else {
				c.NoError(err)
			}

			cost := gorgonia.Must(gorgonia.Mean(y.Output))
			_, err = gorgonia.Grad(cost, input)
			c.NoError(err)

			vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(tn.Learnables()...))
			c.NoError(vm.RunAll())

			tn.PrintWatchables()

			c.Equal(tcase.expectedShape, y.Shape())
			c.Equal(tcase.expectedOutput, y.Value().Data().([]float32))

			yGrad, err := y.Output.Grad()
			c.NoError(err)

			c.Equal(tcase.expectedGrad, yGrad.Data())
			c.Equal(tcase.expectedCost, cost.Value().Data())
		})
	}
}
