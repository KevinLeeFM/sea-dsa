#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <cstdlib>
#include <cstdio>
#include <string>

// Capture argv[0] for death-test reexecs.
static std::string gProgramPath;

// Defined in test translation units to dispatch death scenarios.
bool seadsa_maybeRunDeathScenario(const char *flagName);

// Helper: run the current test binary in a subprocess with a flag to trigger
// an intentional failure/assertion. Returns the subprocess exit code.
static int runSubprocessExpectFailure(const char *flagName) {
	std::string cmd = "SEADSA_DEATH=";
	cmd += flagName;
	cmd += " ";
	cmd += gProgramPath;
	return std::system(cmd.c_str());
}

// Expose to tests.
int seadsa_runSubprocessExpectFailure(const char *flagName) {
	return runSubprocessExpectFailure(flagName);
}

int main(int argc, char **argv) {
	gProgramPath = argv[0] ? argv[0] : "";

	if (const char *deathFlag = std::getenv("SEADSA_DEATH")) {
		if (!seadsa_maybeRunDeathScenario(deathFlag)) {
			std::fprintf(stderr, "Unknown SEADSA_DEATH scenario: %s\n", deathFlag);
			return 1;
		}
		return 0;
	}

	doctest::Context ctx;
	ctx.applyCommandLine(argc, argv);
	return ctx.run();
}
