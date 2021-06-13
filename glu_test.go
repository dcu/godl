package godl

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestGLU(t *testing.T) {
	testCases := []struct {
		desc           string
		input          tensor.Tensor
		vbs            int
		output         int
		expectedShape  tensor.Shape
		expectedErr    string
		expectedOutput []float32
		fcOpts         *FCOpts
	}{
		{
			desc: "Example 1",
			input: tensor.New(
				tensor.WithShape(6, 2),
				tensor.WithBacking([]float32{0.1, -0.5, 0.3, 0.9, 0.04, -0.3, 0.01, 0.09, -0.1, 0.9, 0.7, 0.04}),
			),
			vbs:            2,
			output:         5,
			expectedShape:  tensor.Shape{6, 5},
			expectedOutput: []float32{-0.26894087, -0.26894087, -0.26894087, -0.26894087, -0.26894087, 0.7310513, 0.7310513, 0.7310513, 0.7310513, 0.7310513, -0.26893026, -0.26893026, -0.26893026, -0.26893026, -0.26893026, 0.73091555, 0.73091555, 0.73091555, 0.73091555, 0.73091555, 0.72595197, 0.72595197, 0.72595197, 0.72595197, 0.72595197, -0.26853833, -0.26853833, -0.26853833, -0.26853833, -0.26853833},
		},
		{
			desc: "Example 2",
			input: tensor.New(
				tensor.WithShape(6, 2),
				tensor.WithBacking([]float32{0.1, -0.5, 0.3, 0.9, 0.04, -0.3, 0.01, 0.09, -0.1, 0.9, 0.7, 0.04}),
			),
			vbs:            2,
			output:         5,
			expectedShape:  tensor.Shape{6, 5},
			expectedOutput: []float32{-0.26894087, -0.26894087, -0.26894087, -0.26894087, -0.26894087, 0.7310513, 0.7310513, 0.7310513, 0.7310513, 0.7310513, -0.26893026, -0.26893026, -0.26893026, -0.26893026, -0.26893026, 0.73091555, 0.73091555, 0.73091555, 0.73091555, 0.73091555, 0.72595197, 0.72595197, 0.72595197, 0.72595197, 0.72595197, -0.26853833, -0.26853833, -0.26853833, -0.26853833, -0.26853833},
		},
		{
			desc: "Example 3",
			input: tensor.New(
				tensor.WithShape(6, 2),
				tensor.WithBacking([]float32{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.4}),
			),
			vbs:    2,
			output: 16,
			fcOpts: &FCOpts{
				InputDimension:  2,
				OutputDimension: 32,
				WithBias:        false,
				WeightsInit:     initDummyWeights,
			},
			expectedShape:  tensor.Shape{6, 16},
			expectedOutput: []float32{-0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, 0.73105735, 0.73105735, 0.73105735, 0.73105735, 0.73105735, 0.73105735, 0.73105735, 0.73105735, 0.73105735, 0.73105735, 0.73105735, 0.73105735, 0.73105735, 0.73105735, 0.73105735, 0.73105735, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, -0.2689413, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747, 0.73105747},
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			tn := NewModel()
			tn.Training = true
			g := tn.ExprGraph()

			input := gorgonia.NewTensor(g, tensor.Float32, tcase.input.Shape().Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithName("input"), gorgonia.WithValue(tcase.input))

			var fc Layer
			if tcase.fcOpts != nil {
				fc = FC(tn, *tcase.fcOpts)
			}

			x, err := GLU(tn, GLUOpts{
				InputDimension:   tcase.vbs,
				OutputDimension:  tcase.output,
				WeightsInit:      initDummyWeights,
				VirtualBatchSize: 2,
				FC:               fc,
			})(input)

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			} else {
				c.NoError(err)
			}

			vm := gorgonia.NewTapeMachine(g)
			c.NoError(vm.RunAll())

			c.Equal(tcase.expectedShape, x.Shape())
			c.Equal(tcase.expectedOutput, x.Value().Data().([]float32))
		})
	}
}
