package tabnet

import (
	"log"
	"os"

	"gorgonia.org/tensor"
)

var (
	testLogger *log.Logger
)

func init() {
	f, err := os.Create("test.log")
	if err != nil {
		panic(err)
	}

	testLogger = log.New(f, "[G]", log.LstdFlags)
}

func initDummyWeights(dt tensor.Dtype, s ...int) interface{} {
	v := make([]float64, tensor.Shape(s).TotalSize())

	for i := range v {
		v[i] = 1.0
	}

	return v
}
