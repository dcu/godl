package tabnet

import (
	"log"
	"testing"

	"github.com/dcu/godl"
	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestTabNetNoEmbeddings(t *testing.T) {
	testCases := []struct {
		desc              string
		input             tensor.Tensor
		vbs               int
		independentBlocks int
		sharedBlocks      int
		output            int
		steps             int
		gamma             float64
		epsilon           float64
		momentum          float64
		prediction        int
		attentive         int
		expectedShape     tensor.Shape
		expectedErr       string
		expectedOutput    []float32
		expectedCost      float32
		expectedAcumLoss  float32
	}{
		// {
		// 	desc: "Example 1",
		// 	input: tensor.New(
		// 		tensor.WithShape(4, 4),
		// 		tensor.WithBacking([]float32{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4}),
		// 	),
		// 	vbs:               2,
		// 	output:            12,
		// 	independentBlocks: 2,
		// 	sharedBlocks:      2,
		//  epsilon: 1e-10,
		// 	steps:             5,
		// 	gamma:             1.2,
		//  momentum: 0.02,
		// 	prediction:        64,
		// 	attentive:         64,
		// 	expectedShape:     tensor.Shape{4, 12},
		// 	expectedOutput:    []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638, 447.8060947055638},
		// 	expectedGrad:      []float32{0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332},
		// 	expectedCost:      223.9030473527819,
		// 	expectedAcumLoss:  -1.3862943607198905,
		// },
		{
			desc: "Example 2",
			input: tensor.New(
				tensor.WithShape(4, 5),
				tensor.WithBacking([]float32{0.4, 1.4, 2.4, 1, 1, 4.4, 5.4, 6.4, 1, 1, 8.4, 9.4, 10.4, 1, 1, 12.4, 13.4, 14.4, 1, 1}),
			),
			vbs:               128,
			output:            1,
			independentBlocks: 2,
			sharedBlocks:      2,
			steps:             3,
			gamma:             1.3,
			epsilon:           1e-15,
			momentum:          0.02,
			prediction:        8,
			attentive:         8,
			expectedShape:     tensor.Shape{4, 1},
			expectedOutput:    []float32{0, 0, 0.4864648, 61.30994},
			expectedCost:      15.449101,
			expectedAcumLoss:  -1.6094379124340954,
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			tn := godl.NewModel()

			g := tn.ExprGraph()

			x := gorgonia.NewTensor(g, tensor.Float32, 2, gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithName("Input"), gorgonia.WithValue(tcase.input))

			a := gorgonia.NewTensor(g, tensor.Float32, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithInit(gorgonia.Ones()), gorgonia.WithName("AttentiveX"))
			priors := gorgonia.NewTensor(g, tensor.Float32, tcase.input.Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithInit(gorgonia.Ones()), gorgonia.WithName("Priors"))

			result, err := TabNetNoEmbeddings(tn, TabNetNoEmbeddingsOpts{
				VirtualBatchSize:   tcase.vbs,
				IndependentBlocks:  tcase.independentBlocks,
				PredictionLayerDim: tcase.prediction,
				AttentionLayerDim:  tcase.attentive,
				OutputSize:         tcase.output,
				SharedBlocks:       tcase.sharedBlocks,
				DecisionSteps:      tcase.steps,
				Gamma:              tcase.gamma,
				InputSize:          a.Shape()[1],
				BatchSize:          a.Shape()[0],
				WeightsInit:        initDummyWeights,
				ScaleInit:          gorgonia.Ones(),
				BiasInit:           gorgonia.Zeroes(),
				Epsilon:            tcase.epsilon,
				Momentum:           tcase.momentum,
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
			_, err = gorgonia.Grad(cost, append(tn.Learnables(), x)...)
			c.NoError(err)

			vm := gorgonia.NewTapeMachine(g,
				gorgonia.BindDualValues(tn.Learnables()...),
				gorgonia.WithLogger(testLogger),
				gorgonia.WithValueFmt("%+v"),
				gorgonia.WithWatchlist(),
			)
			c.NoError(vm.RunAll())

			tn.PrintWatchables()
			// fmt.Printf("%v\n", g.String())

			log.Printf("input grad: %v", x.Deriv().Value())

			c.Equal(tcase.expectedShape, y.Shape())
			c.Equal(tcase.expectedOutput, y.Value().Data().([]float32))

			c.Equal(tcase.expectedCost, cost.Value().Data())
			c.Equal(tcase.expectedAcumLoss, result.Loss.Value().Data())

			w := tn.Learnables()[len(tn.Learnables())-1]

			optim := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.02))
			err = optim.Step([]gorgonia.ValueGrad{w})
			c.NoError(err)

			log.Printf("weight updated: %v\n\n\n", w.Value())
		})
	}
}
