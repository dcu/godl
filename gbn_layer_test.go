package tabnet

import (
	"testing"

	"github.com/stretchr/testify/require"
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestGBN(t *testing.T) {
	g := NewGraph()

	testCases := []struct {
		desc  string
		input *Node
	}{
		{
			desc:  "example 1",
			input: NewMatrix(g, tensor.Float64, WithShape(100, 50), WithName("input"), WithInit(GlorotN(1.0))),
		},
	}

	tn := &TabNet{}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)
			x, err := tn.GBN(tcase.input, 10, 0.0, 0.0)

			c.NoError(err)
			c.Equal(tensor.Shape{100, 50}, x.Shape())
		})
	}
}
