package mnist

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLoadData(t *testing.T) {
	testCases := []struct {
		desc     string
		mode     Mode
		examples int
	}{
		{
			desc:     "ModeTrain",
			mode:     ModeTrain,
			examples: 600000,
		},
		{
			desc:     "ModeTest",
			mode:     ModeTest,
			examples: 100000,
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			c := require.New(t)

			x, y, err := Load(tC.mode, "")
			c.NoError(err)
			c.NotNil(x)
			c.NotNil(y)

			c.Equal(x.Shape()[0], y.Shape()[0])

			c.Equal(tC.examples, y.Size())
			c.Equal(tC.examples*28*28/10, x.Size())
		})
	}
}
