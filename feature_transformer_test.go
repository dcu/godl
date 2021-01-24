package tabnet

import (
	"testing"

	"github.com/stretchr/testify/require"
	. "gorgonia.org/gorgonia"
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
		expectedOutput    []float64
	}{
		{
			desc: "Example1",
			input: tensor.New(
				tensor.WithShape(6, 2),
				tensor.WithBacking([]float64{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.4}),
			),
			vbs:               2,
			output:            2,
			independentBlocks: 5,
			sharedBlocks:      0,
			expectedShape:     tensor.Shape{6, 2},
			expectedOutput:    []float64{-0.46379327241699114, -0.2870165771203542, 1.87719497530188, 2.053971670598517, 0.24331350876955662, 0.42009020406619374, 2.5843017564884283, 2.761078451785065, 0.9504202899561044, 1.1271969852527415, 3.291408537674975, 3.468185232971612},
		},
		{
			desc: "Example2",
			input: tensor.New(
				tensor.WithShape(6, 2),
				tensor.WithBacking([]float64{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.4}),
			),
			vbs:               2,
			output:            16,
			independentBlocks: 5,
			sharedBlocks:      5,
			expectedShape:     tensor.Shape{6, 16},
			expectedOutput:    []float64{-0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, -0.6324731811198426, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826, 1.719240284475826},
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			g := NewGraph()
			tn := &Model{g: g}

			input := NewTensor(g, tensor.Float64, tcase.input.Dims(), WithShape(tcase.input.Shape()...), WithName("input"), WithValue(tcase.input))

			shared := make([]Layer, tcase.sharedBlocks)
			fcInput := input.Shape()[1]
			fcOutput := 2 * (8 + 8)
			for i := 0; i < tcase.sharedBlocks; i++ {
				shared[i] = tn.FC(FCOpts{
					OutputDimension: fcOutput, // double the size so we can take half and half
					WeightsInit:     initDummyWeights,
					InputDimension:  fcInput,
				})

				fcInput = 8 + 8
			}

			x, _, err := tn.FeatureTransformer(FeatureTransformerOpts{
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

			vm := NewTapeMachine(g)
			c.NoError(vm.RunAll())

			c.Equal(tcase.expectedShape, x.Shape())
			c.Equal(tcase.expectedOutput, x.Value().Data().([]float64))
		})
	}
}
