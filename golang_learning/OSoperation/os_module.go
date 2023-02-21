package OSoperation

import (
	"fmt"
	"log"
	"os"
)

func CreateFile() {
	file, err := os.Create("test.md")
	if err != nil {
		fmt.Println("err: ", err)
	} else {
		fmt.Printf("file.Name():%v\n", file.Name())
	}
}
func MkDir() {
	err := os.Mkdir("test", os.ModePerm)
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}
}
func Remove_file() {
	err := os.Remove("test.md")
	if err != nil {
		fmt.Printf("err: %v\n", err) // err msg: no such file or directory
	}
}
func ReadFile() {
	b, err := os.ReadFile("go.mod")
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("b: %v\n", string(b[:]))
}
func WriteFile() {
	err := os.WriteFile("test.md", []byte("README FIRST\n"), os.ModePerm)
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}
}
func Open_file_append() {
	f, err := os.OpenFile("test.md", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatal(err)
	}
	if _, err := f.Write([]byte("This setence is appended\n")); err != nil {
		f.Close()
		log.Fatal(err)
	}
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
}
