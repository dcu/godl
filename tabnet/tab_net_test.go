package tabnet

import (
	"log"
	"os"
	"testing"

	"github.com/dcu/godl"
	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestTabNetEmbeddings(t *testing.T) {
	testCases := []struct {
		desc              string
		input             tensor.Tensor
		vbs               int
		independentBlocks int
		sharedBlocks      int
		output            int
		steps             int
		gamma             float64
		prediction        int
		attentive         int
		expectedShape     tensor.Shape
		expectedErr       string
		expectedOutput    []float32
		expectedGrad      []float32
		expectedCost      float32
		expectedAcumLoss  float32
	}{
		{
			desc: "Example 1",
			input: tensor.New(
				tensor.WithShape(4, 4),
				tensor.WithBacking([]float32{0.4, 1.4, 2.4, 0, 4.4, 5.4, 6.4, 1, 8.4, 9.4, 10.4, 2, 12.4, 13.4, 14.4, 3}),
			),
			vbs:               4,
			output:            12,
			independentBlocks: 2,
			sharedBlocks:      2,
			steps:             5,
			gamma:             1.2,
			prediction:        8,
			attentive:         8,
			expectedShape:     tensor.Shape{4, 12},
			expectedOutput:    []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.81077474, 0.81077474, 0.81077474, 0.81077474, 0.81077474, 0.81077474, 0.81077474, 0.81077474, 0.81077474, 0.81077474, 0.81077474, 0.81077474, 102.183235, 102.183235, 102.183235, 102.183235, 102.183235, 102.183235, 102.183235, 102.183235, 102.183235, 102.183235, 102.183235, 102.183235},
			expectedGrad:      []float32{0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332},
			expectedCost:      25.748503,
			expectedAcumLoss:  -1.609438,
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			tn := godl.NewModel()
			logFile, _ := os.Create("tabnet.log")
			defer func() { _ = logFile.Close() }()

			tn.Logger = log.New(logFile, "", log.LstdFlags)

			g := tn.TrainGraph()

			x := gorgonia.NewTensor(g, tensor.Float32, 2, gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithName("Input"), gorgonia.WithValue(tcase.input))

			a := gorgonia.NewTensor(g, tensor.Float32, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithInit(gorgonia.Ones()), gorgonia.WithName("AttentiveX"))
			priors := gorgonia.NewTensor(g, tensor.Float32, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithInit(gorgonia.Ones()), gorgonia.WithName("Priors"))

			result, err := TabNet(tn, TabNetOpts{
				VirtualBatchSize:   tcase.vbs,
				IndependentBlocks:  tcase.independentBlocks,
				PredictionLayerDim: tcase.prediction,
				AttentionLayerDim:  tcase.attentive,
				OutputSize:         tcase.output,
				SharedBlocks:       tcase.sharedBlocks,
				DecisionSteps:      tcase.steps,
				Gamma:              tcase.gamma,
				InputSize:          a.Shape()[0],
				BatchSize:          a.Shape()[0],
				WeightsInit:        initDummyWeights,
				ScaleInit:          gorgonia.Ones(),
				BiasInit:           gorgonia.Zeroes(),
				Epsilon:            1e-10,
				CatIdxs:            []int{3},
				CatDims:            []int{4},
				CatEmbDim:          []int{2},
			})(x, a, priors)

			y := result.Output

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			} else {
				c.NoError(err)
			}

			cost := gorgonia.Must(gorgonia.Mean(y))
			_, err = gorgonia.Grad(cost, append([]*gorgonia.Node{x}, tn.Learnables()...)...)
			c.NoError(err)

			optimizer := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.02))

			vm := gorgonia.NewTapeMachine(g,
				gorgonia.BindDualValues(tn.Learnables()...),
				gorgonia.WithLogger(testLogger),
				gorgonia.WithValueFmt("%+v"),
				gorgonia.WithWatchlist(),
				gorgonia.WithNaNWatch(),
				gorgonia.WithInfWatch(),
			)

			err = vm.RunAll()
			tn.PrintWatchables()
			c.NoError(err)

			err = optimizer.Step(gorgonia.NodesToValueGrads(tn.Learnables()))
			c.NoError(err)

			vm.Reset()

			// fmt.Printf("%v\n", g.String())

			log.Printf("input grad: %v", x.Deriv().Value())

			c.Equal(tcase.expectedShape, y.Shape())

			log.Printf("[train] y: %#v", y.Value().Data())
			log.Printf("[train] cost: %#v", cost.Value().Data())
			log.Printf("[train] accum lost: %#v", result.Loss.Value().Data())

			c.InDeltaSlice(tcase.expectedOutput, y.Value().Data().([]float32), 1e-5)

			c.InDelta(tcase.expectedCost, cost.Value().Data(), 1e-5)
			c.Equal(tcase.expectedAcumLoss, result.Loss.Value().Data())

			vmEval := gorgonia.NewTapeMachine(g,
				gorgonia.EvalMode(),
				gorgonia.WithLogger(testLogger),
				gorgonia.WithValueFmt("%+v"),
				gorgonia.WithWatchlist(),
				gorgonia.WithNaNWatch(),
				gorgonia.WithInfWatch(),
			)

			err = vmEval.RunAll()
			tn.PrintWatchables()
			c.NoError(err)

			log.Printf("[eval] y: %#v", y.Value().Data())
			log.Printf("[eval] accum lost: %#v", result.Loss.Value().Data())
		})
	}
}
