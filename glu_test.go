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
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			tn := NewModel()
			tn.Training = true
			g := tn.ExprGraph()

			input := gorgonia.NewTensor(g, tensor.Float32, tcase.input.Shape().Dims(), gorgonia.WithShape(tcase.input.Shape()...), gorgonia.WithName("input"), gorgonia.WithValue(tcase.input))

			x, err := GLU(tn, GLUOpts{
				InputDimension:   tcase.vbs,
				OutputDimension:  tcase.output,
				WeightsInit:      initDummyWeights,
				VirtualBatchSize: 2,
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
