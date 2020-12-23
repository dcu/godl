package tabnet

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestTabNet(t *testing.T) {
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
			desc: "Example 1",
			input: NewTensor(g, tensor.Float64, 2, WithShape(12, 1), WithName("Input"), WithValue(
				tensor.New(
					tensor.WithShape(12, 1),
					tensor.WithBacking([]float64{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.4}),
				),
			)),
			vbs:               2,
			output:            2,
			independentBlocks: 2,
			expectedShape:     tensor.Shape{12, 12},
			expectedOutput:    []float64{},
		},
	}

	tn := &Model{g: g}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			a := NewTensor(g, Float64, tcase.input.Dims(), WithShape(tcase.input.Shape()...), WithInit(Ones()), WithName("AttentiveX"))
			priors := NewTensor(g, Float64, tcase.input.Dims(), WithShape(tcase.input.Shape()...), WithInit(Ones()), WithName("Priors"))

			x, err := tn.TabNet(TabNetOpts{
				VirtualBatchSize:   tcase.vbs,
				IndependentBlocks:  tcase.independentBlocks,
				PredictionLayerDim: 64,
				AttentionLayerDim:  64,
				WeightsInit:        initDummyWeights,
				OutputFeatures:     12,
				SharedBlocks:       2,
				DecisionSteps:      5,
				Gamma:              1.2,
				InputDim:           1,
				Inferring:          false,
			})(tcase.input, a, priors)

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			} else {
				c.NoError(err)
			}

			vm := NewTapeMachine(g,
				gorgonia.BindDualValues(tn.learnables...),
				gorgonia.WithLogger(testLogger),
				gorgonia.WithValueFmt("%+v"),
				gorgonia.WithWatchlist(),
			)
			c.NoError(vm.RunAll())

			c.Equal(tcase.expectedShape, x.Shape())
			c.Equal(tcase.expectedOutput, x.Value().Data().([]float64))
		})
	}
}
