package storage

import (
	"fmt"
	"log"
	"path/filepath"

	"gorgonia.org/tensor"
)

// Storage is in charge of loading the weights from files
type Storage struct {
	cost       float64
	learnables map[string]Weight
}

// NewStorage instantiates a storage
func NewStorage() *Storage {
	return &Storage{
		cost:       0.0,
		learnables: map[string]Weight{},
	}
}

// TensorByName returns the tensor associated to a weight name
func (l *Storage) TensorByName(name string) (tensor.Tensor, error) {
	t, ok := l.learnables[name]
	if !ok {
		return nil, ErrLearnableNotFound
	}

	return t.Value.(tensor.Tensor), nil
}

// Load loads the weights in the given path
func (l *Storage) LoadFile(filePath string) error {
	ext := filepath.Ext(filePath)

	switch ext {
	case ".nn1":
		return LoadNN1(l, filePath)
	default:
		return fmt.Errorf("extension %v is not supported yet", ext)
	}
}

// AddWeights adds weights to the storage
func (l *Storage) AddWeights(weights ...Weight) {
	for _, w := range weights {
		if _, ok := l.learnables[w.Name]; ok {
			log.Panicf("weight %s is already present in the storage", w.Name)
		}

		l.learnables[w.Name] = w
	}
}
