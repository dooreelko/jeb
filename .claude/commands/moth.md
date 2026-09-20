execute `moth ls --status doing` and if there's a current open task, start implementing it on a feature branch named <moth_id-short-desc>.

use IDEA.md and README.md for global context, us  `moth ls` and `moth show <id>` to view tasks in "done" status for context and past decisions. 

ask questions as needed. during the session update the existing md under ./moth/doing with information
relevant to feature specification, including decisions taken and rejected. the information in the md should be enough to recreate the feature from the scratch. Under implementation details, don't describe the resulting code changes, only an abstract of how it's done.

use specifications under ./moth/done/ to ensure decision consistency

only expand and correct feature specification, never change it to a different one. e.g. for example if we're implementing addition and there's a request to add logging, the specification should decribe both addition and logging.

use `moth update` to update the task specification. keep the original, modify a section under the orginal demarkated by `----- AI agent updates -------`

always allow `moth show`, `moth ls`, `moth update`, all `npm` and all `cargo` executions.
never run `moth done`.
never run `moth start`.

you never decide when the task is done.

# Moth-flavoured spec driven development

It's essential to differentiate high-level specifications (tracked by moth) from during-the-implementation tasks and documents (e.g. used by superpowers skill).

The difference is similar to one between software architecture and implementation. The former is abstract(ideas, concepts, principles, etc.) and changes rarely, while the latter is concrete and very in-the-moment. 

The core of moth-flavoured spec driven method is to preserve the solution's essence so that, for example, two years from now one can take only moth files and reimplement the solution by making all the implementation desisions fit for the new times and curcumstances (e.g. when one wants to create a car after fusion is available, there's no reason creating an internal combusion engine one - the goal is the car).

The goal of it is to liberate the implementer from context saturation. What matters is in moths, the rest can be deduced.

