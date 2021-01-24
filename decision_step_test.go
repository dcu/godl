package tabnet

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestDecisionStep(t *testing.T) {
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
		expectedPriors    []float64
	}{
		{
			desc: "Example 1",
			input: NewTensor(g, tensor.Float64, 2, WithShape(4, 7), WithName("input"), WithValue(
				tensor.New(
					tensor.WithShape(4, 7),
					tensor.WithBacking([]float64{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 0.4, 1.4}),
				),
			)),
			vbs:               2,
			independentBlocks: 3,
			expectedShape:     tensor.Shape{4, 7},
			expectedOutput:    []float64{-0.27830426180944234, 0.07524912878383146, 0.42880251937710523, 0.7823559099703792, 1.1359093005636534, 1.4894626911569269, 1.8430160817502006, 3.757222084193693, 1.2823483500407766, 1.6359017406340506, 1.989455131227324, 2.343008521820598, 2.6965619124138724, 3.0501153030071455, 3.4036753862990436, 3.7572287768923176, 4.110782167485591, 4.464335558078865, 1.2823550427394004, 1.635908433332674, 1.989461823925948, 0.7823553881246798, 1.1359087787179536, 1.4894621693112278, 1.8430155599045015, 2.1965689504977752, -0.2783047836551418, 0.07524860693813204},
			expectedPriors:    []float64{1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572, 1.0571428571428572},
		},
	}

	tn := &Model{g: g}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			a := NewTensor(g, Float64, tcase.input.Dims(), WithShape(tcase.input.Shape()...), WithInit(Ones()))
			priors := NewTensor(g, Float64, tcase.input.Dims(), WithShape(tcase.input.Shape()...), WithInit(Ones()))
			step := tn.DecisionStep(DecisionStepOpts{
				VirtualBatchSize:   tcase.vbs,
				Shared:             nil,
				IndependentBlocks:  tcase.independentBlocks,
				PredictionLayerDim: 7,
				AttentionLayerDim:  0,
				WeightsInit:        initDummyWeights,
				InputDimension:     a.Shape()[0],
				OutputDimension:    a.Shape()[1],
			})

			mask, _, err := step.AttentiveTransformer(a, priors)
			c.NoError(err)

			// Update prior
			{
				gamma := gorgonia.NewScalar(tn.g, tensor.Float64, gorgonia.WithValue(1.2))
				priors, err = gorgonia.HadamardProd(priors, gorgonia.Must(gorgonia.Sub(gamma, mask)))
				c.NoError(err)
			}

			ds, _, err := step.FeatureTransformer(tcase.input)
			c.NoError(err)

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			} else {
				c.NoError(err)
			}

			vm := NewTapeMachine(g,
				gorgonia.WithLogger(testLogger),
				gorgonia.WithValueFmt("%v"),
				gorgonia.TraceExec(),
			)
			c.NoError(vm.RunAll())

			c.Equal(tcase.expectedShape, ds.Shape())
			c.Equal(tcase.expectedOutput, ds.Value().Data().([]float64))

			c.Equal(tcase.expectedShape, priors.Shape())
			c.Equal(tcase.expectedPriors, priors.Value().Data().([]float64))
		})
	}
}
