import sys

from sgx_description.utils.config import SUPABASE_CLIENT
from sgx_description.generate_description import open_json


def update_description(path: str, column_name: str):
    payload = open_json(
       path=path
    )

    if payload is None:
        print(f'no payload found at {path} for: {column_name}, skipping')
        return

    for sector, description in payload.items():
        (
            SUPABASE_CLIENT
            .table("sgx_sector_metadata")
            .update({
                column_name: description
            })
            .eq("sector", sector)
            .execute()
        )

    print(f'upserted: {len(payload)} for: {column_name}')


DESCRIPTION_TARGETS = {
    'description': {
        'path': 'sgx_description/data/description/generated_descriptions.json',
        'column_name': 'description',
    },
    'growth': {
        'path': 'sgx_description/data/growth_description/generated_description.json',
        'column_name': 'growth_index_description',
    },
    'pe': {
        'path': 'sgx_description/data/pe_description/generated_description.json',
        'column_name': 'pe_index_description',
    },
    'health': {
        'path': 'sgx_description/data/health_description/generated_description.json',
        'column_name': 'health_index_description',
    },
}


if __name__ == '__main__':
    keys = sys.argv[1:] or DESCRIPTION_TARGETS.keys()

    for key in keys:
        update_description(**DESCRIPTION_TARGETS[key])


# uv run -m sgx_description.upsert_description

