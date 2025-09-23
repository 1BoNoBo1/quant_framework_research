"""
Database Migrations
===================

This directory contains database migrations for the QFrame system.
Each migration file should define a 'migration' variable that is an instance
of Migration, SQLMigration, or PythonMigration.

Example SQL Migration:
----------------------
from ..migrations import SQLMigration

up_sql = "CREATE TABLE example (id UUID PRIMARY KEY);"
down_sql = "DROP TABLE example;"

migration = SQLMigration(
    version="002",
    name="add_example_table",
    up_sql=up_sql,
    down_sql=down_sql,
    description="Add example table"
)

Example Python Migration:
------------------------
from ..migrations import PythonMigration

async def upgrade(db_manager):
    # Custom upgrade logic
    await db_manager.execute_query("...")

async def downgrade(db_manager):
    # Custom downgrade logic
    await db_manager.execute_query("...")

migration = PythonMigration(
    version="003",
    name="custom_migration",
    up_func=upgrade,
    down_func=downgrade,
    description="Custom migration with Python code"
)
"""
