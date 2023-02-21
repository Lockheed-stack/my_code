package LogTest

import (
	"fmt"
	"log"
	"os"
)

func LogPrint() {
	log.Print()
}
func PanicTest() {
	defer fmt.Println("exec defer after panic")
	log.Printf("log print...")
	log.Panic("it's log panic\n")
	fmt.Println("I will not be executed")
}
func FatalTest() {
	defer fmt.Println("defer will not exce after Fatal")
	log.Println("Fatal will happen soon...")
	log.Fatal("Fatal happened!")
}
func SetLogFlag() {
	log.SetFlags(log.LUTC | log.Ldate | log.Lshortfile)
	log.Print("after set flags...")
}
func WriteLogToFile() {
	f, err := os.OpenFile("test.md", os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		log.Fatal("open test.md error")
	}

	log.SetOutput(f)
	SetLogFlag()
	LogPrint()
	f.Close()
}
