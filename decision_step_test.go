package tabnet

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestDecisionStep(t *testing.T) {
	g := gorgonia.NewGraph()

	testCases := []struct {
		desc              string
		input             *gorgonia.Node
		vbs               int
		independentBlocks int
		output            int
		expectedShape     tensor.Shape
		expectedErr       string
		expectedOutput    []float32
		expectedPriors    []float32
	}{
		{
			desc: "Example 1",
			input: gorgonia.NewTensor(g, tensor.Float32, 2, gorgonia.WithShape(4, 7), gorgonia.WithName("input"), gorgonia.WithValue(
				tensor.New(
					tensor.WithShape(4, 7),
					tensor.WithBacking([]float32{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 0.4, 1.4}),
				),
			)),
			vbs:               2,
			independentBlocks: 3,
			expectedShape:     tensor.Shape{4, 7},
			expectedOutput:    []float32{-0.27830428, 0.075249106, 0.42880255, 0.78235584, 1.1359093, 1.4894627, 1.843016, 3.7572215, 1.2823482, 1.6359016, 1.989455, 2.3430083, 2.6965618, 3.050115, 3.4036748, 3.7572284, 4.1107817, 4.464335, 1.2823547, 1.6359084, 1.9894617, 0.7823554, 1.1359088, 1.489462, 1.8430156, 2.196569, -0.27830482, 0.07524856},
			expectedPriors:    []float32{1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572},
		},
	}

	tn := &Model{g: g}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			a := gorgonia.NewTensor(g, gorgonia.Float32, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithInit(gorgonia.Ones()))
			priors := gorgonia.NewTensor(g, gorgonia.Float32, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithInit(gorgonia.Ones()))
			step := NewDecisionStep(tn, DecisionStepOpts{
				VirtualBatchSize:   tcase.vbs,
				Shared:             nil,
				IndependentBlocks:  tcase.independentBlocks,
				PredictionLayerDim: 7,
				AttentionLayerDim:  0,
				WeightsInit:        initDummyWeights,
				InputDimension:     a.Shape()[0],
				OutputDimension:    a.Shape()[1],
			})

			mask, err := step.AttentiveTransformer(a, priors)
			c.NoError(err)

			// Update prior
			{
				gamma := gorgonia.NewScalar(tn.g, tensor.Float32, gorgonia.WithValue(float32(1.2)))
				priors, err = gorgonia.HadamardProd(priors, gorgonia.Must(gorgonia.Sub(gamma, mask.Output)))
				c.NoError(err)
			}

			ds, err := step.FeatureTransformer(tcase.input)
			c.NoError(err)

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			} else {
				c.NoError(err)
			}

			vm := gorgonia.NewTapeMachine(g,
				gorgonia.WithLogger(testLogger),
				gorgonia.WithValueFmt("%v"),
				gorgonia.TraceExec(),
			)
			c.NoError(vm.RunAll())

			c.Equal(tcase.expectedShape, ds.Shape())
			c.Equal(tcase.expectedOutput, ds.Value().Data().([]float32))

			c.Equal(tcase.expectedShape, priors.Shape())
			c.Equal(tcase.expectedPriors, priors.Value().Data().([]float32))
		})
	}
}
