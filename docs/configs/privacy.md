# Model Privacy

stdllm supports various privacy levels for models, allowing you to choose the appropriate level of data protection for your use case.

## Privacy Levels

| Level | Description |
|-------|-------------|
| `BASIC` | Standard privacy level with default provider protections. Data may be used for model improvements. Suitable for non-sensitive information. |
| `HIPAA` | HIPAA-compliant processing for healthcare data. Includes strict access controls, audit trails, and data encryption. |
| `GDPR` | Compliant with EU General Data Protection Regulation. Ensures data sovereignty, processing limitations, and user rights protection. |
| `FEDRAMP` | Federal Risk and Authorization Management Program certified. Meets US government security standards for cloud services. |
| `SOC2` | Service Organization Control 2 certified. Ensures security, availability, processing integrity, and data confidentiality. |

## Filtering Models by Privacy Level

You can list models that match specific privacy requirements using the registry:

```python
from stdllm.config.metadata import PrivacyLevel
from stdllm.registry import ModelRegistry, SetFilter

registry = ModelRegistry()

# List models with any of the specified privacy levels
hipaa_or_gdpr_models = list(registry.list_models(
    privacy_levels={PrivacyLevel.HIPAA, PrivacyLevel.GDPR}
))

# List models that have ALL specified privacy levels
fully_compliant_models = list(registry.list_models(
    privacy_levels=SetFilter(
        values={PrivacyLevel.HIPAA, PrivacyLevel.GDPR},
        mode="all"
    )
))
```


!!! note "Multiple Privacy Levels"
    Models can support multiple privacy levels simultaneously. For example, a model might be both HIPAA and GDPR compliant.

!!! warning "Privacy Verification"
    Privacy levels are based on provider claims and certifications. Always verify the actual compliance requirements for your specific use case.
