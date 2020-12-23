package tabnet

import (
	"testing"

	"github.com/stretchr/testify/require"
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestFeatureTransformer(t *testing.T) {
	g := NewGraph()

	testCases := []struct {
		desc              string
		input             *Node
		vbs               int
		independentBlocks int
		output            int
		expectedShape     tensor.Shape
		expectedErr       string
		expectedOutput    []float64
	}{
		{
			desc: "",
			input: NewTensor(g, tensor.Float64, 2, WithShape(6, 2), WithName("input"), WithValue(
				tensor.New(
					tensor.WithShape(6, 2),
					tensor.WithBacking([]float64{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.4}),
				),
			)),
			vbs:               2,
			output:            2,
			independentBlocks: 5,
			expectedShape:     tensor.Shape{6, 2},
			expectedOutput:    []float64{-0.4395035730343544, -0.28961441959885387, 3.3431244714105195, 0.7463088298910071, 0.8543397630255111, 0.33027296796752753, 4.136399173999854, 0.976768932989377, 2.0747412532576526, 0.995551329231948, 4.497775695459858, 1.494460903074312},
		},
	}

	tn := &Model{g: g}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			priors := NewTensor(g, Float64, tcase.input.Dims(), WithShape(tcase.input.Shape()...), WithInit(Ones()))
			x, err := tn.FeatureTransformer(FeatureTransformerOpts{
				VirtualBatchSize:  tcase.vbs,
				OutputFeatures:    tcase.output,
				Shared:            nil,
				IndependentBlocks: tcase.independentBlocks,
				WeightsInit:       initDummyWeights,
			})(tcase.input, priors)

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
