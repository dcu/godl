package tabnet

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestTabNet(t *testing.T) {
	testCases := []struct {
		desc              string
		input             tensor.Tensor
		vbs               int
		independentBlocks int
		sharedBlocks      int
		output            int
		steps             int
		gamma             float32
		prediction        int
		attentive         int
		expectedShape     tensor.Shape
		expectedErr       string
		expectedOutput    []float32
	}{
		{
			desc: "Example 1",
			input: tensor.New(
				tensor.WithShape(4, 4),
				tensor.WithBacking([]float32{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4}),
			),
			vbs:               2,
			output:            12,
			independentBlocks: 2,
			sharedBlocks:      2,
			steps:             5,
			gamma:             1.2,
			prediction:        64,
			attentive:         64,
			expectedShape:     tensor.Shape{4, 12},
			expectedOutput:    []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062, 447.8062},
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			g := gorgonia.NewGraph()
			tn := &Model{g: g}

			x := gorgonia.NewTensor(g, tensor.Float32, 2, gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithName("Input"), gorgonia.WithValue(tcase.input))

			a := gorgonia.NewTensor(g, tensor.Float32, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithInit(gorgonia.Ones()), gorgonia.WithName("AttentiveX"))
			priors := gorgonia.NewTensor(g, tensor.Float32, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithInit(gorgonia.Ones()), gorgonia.WithName("Priors"))

			x, _, err := tn.TabNet(TabNetOpts{
				VirtualBatchSize:   tcase.vbs,
				IndependentBlocks:  tcase.independentBlocks,
				PredictionLayerDim: tcase.prediction,
				AttentionLayerDim:  tcase.attentive,
				OutputDimension:    tcase.output,
				SharedBlocks:       tcase.sharedBlocks,
				DecisionSteps:      tcase.steps,
				Gamma:              tcase.gamma,
				InputDimension:     a.Shape()[0],
				BatchSize:          a.Shape()[0],
				Inferring:          false,
				WeightsInit:        initDummyWeights,
				ScaleInit:          gorgonia.Ones(),
				BiasInit:           gorgonia.Zeroes(),
			})(x, a, priors)

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			} else {
				c.NoError(err)
			}

			vm := gorgonia.NewTapeMachine(g,
				gorgonia.BindDualValues(tn.learnables...),
				gorgonia.WithLogger(testLogger),
				gorgonia.WithValueFmt("%+v"),
				gorgonia.WithWatchlist(),
			)
			c.NoError(vm.RunAll())

			// fmt.Printf("%v\n", g.String())

			c.Equal(tcase.expectedShape, x.Shape())
			c.Equal(tcase.expectedOutput, x.Value().Data().([]float32))
		})
	}
}
