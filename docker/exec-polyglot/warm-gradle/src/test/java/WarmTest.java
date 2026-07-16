import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

// Trivial test whose only job is to make `gradle test` at image-build time download the full
// JUnit5 + AssertJ runtime (launcher, engine, api, assertj) into the gradle cache for offline use.
class WarmTest {
    @Test
    void warmsTheCache() {
        assertThat(1 + 1).isEqualTo(2);
    }
}
